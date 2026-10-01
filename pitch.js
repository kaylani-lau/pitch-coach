// Shared engine for the voice-tools pages: mic capture, pitch detection (YIN),
// a rolling 10-second pitch history, replay of the last 10 seconds, and a graph renderer.
// Runs entirely in the browser. Nothing is uploaded or saved.
(function (global) {
  'use strict';

  // Ranges approximated from the reference app's screenshot. Edit to taste.
  const BANDS = [
    { id: 'female', label: 'Female', lo: 178, hi: 310 },
    { id: 'andro', label: 'Androgynous', lo: 162, hi: 178 },
    { id: 'male', label: 'Male', lo: 85, hi: 162 },
  ];
  const WINDOW_SEC = 10;
  const BLOCK = 2048; // samples per analysis chunk; one history point per chunk
  const PREROLL_SEC = 0.25; // replay starts this long before the first voiced moment
  const FMIN = 60;
  const FMAX = 500;
  const PHRASES = [
    'Can I get your phone number?',
    'Hi, how are you doing today?',
    "I'd like a medium coffee, please.",
    'The weather is really nice today.',
    'Could you tell me where the station is?',
    'Thanks so much, see you later!',
    'What time does the store open tomorrow?',
    'I think we should try the new place downtown.',
  ];

  function bandFor(hz) {
    if (hz == null) return null;
    for (const band of BANDS) {
      if (hz >= band.lo && hz < band.hi) return band;
    }
    return null;
  }

  // Band name for a pitch, or "Below range" / "Above range" outside all bands.
  function labelFor(hz) {
    if (hz == null) return null;
    const band = bandFor(hz);
    if (band) return band.label;
    const lowest = Math.min(...BANDS.map((b) => b.lo));
    return hz < lowest ? 'Below range' : 'Above range';
  }

  // YIN pitch detection. Returns Hz, or null for silence / unvoiced sound.
  function detectPitch(buf, sampleRate, opts) {
    const o = opts || {};
    const fmin = o.fmin || FMIN;
    const fmax = o.fmax || FMAX;
    const threshold = o.threshold || 0.15;
    const minRms = o.minRms == null ? 0.008 : o.minRms;

    let energy = 0;
    for (let i = 0; i < buf.length; i++) energy += buf[i] * buf[i];
    if (Math.sqrt(energy / buf.length) < minRms) return null;

    const maxTau = Math.min(Math.floor(sampleRate / fmin), Math.floor(buf.length / 2));
    const minTau = Math.max(2, Math.floor(sampleRate / fmax));
    const width = buf.length - maxTau;

    // Difference function.
    const d = new Float32Array(maxTau + 1);
    for (let tau = 1; tau <= maxTau; tau++) {
      let sum = 0;
      for (let j = 0; j < width; j++) {
        const diff = buf[j] - buf[j + tau];
        sum += diff * diff;
      }
      d[tau] = sum;
    }

    // Cumulative mean normalized difference.
    d[0] = 1;
    let running = 0;
    for (let tau = 1; tau <= maxTau; tau++) {
      running += d[tau];
      d[tau] = running ? (d[tau] * tau) / running : 1;
    }

    // First dip below the threshold, then walk down to its local minimum.
    let tau = -1;
    for (let t = minTau; t <= maxTau; t++) {
      if (d[t] < threshold) {
        while (t + 1 <= maxTau && d[t + 1] < d[t]) t++;
        tau = t;
        break;
      }
    }
    if (tau < 0) return null;

    // Parabolic interpolation for sub-sample accuracy.
    let refined = tau;
    if (tau > 1 && tau < maxTau) {
      const a = d[tau - 1];
      const b = d[tau];
      const c = d[tau + 1];
      const denom = a + c - 2 * b;
      if (denom) refined = tau + (a - c) / (2 * denom);
    }
    const hz = sampleRate / refined;
    if (hz < fmin || hz > fmax) return null;
    return hz;
  }

  function median(values) {
    if (!values.length) return null;
    const sorted = values.slice().sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  }

  // States: idle (never started), waiting (mic open, but the browser holds audio paused
  // until the page is tapped), live, replay, stopped.
  function createTracker() {
    let ctx = null;
    let stream = null;
    let proc = null;
    let source = null;
    let sampleRate = 48000;
    let ring = null;
    let ringPos = 0;
    let filled = 0;
    let capturing = false;
    let replayNode = null;
    let replayStart = 0;
    let replayDur = 0;
    let frozenNow = 0;
    let starting = null;
    const history = []; // { t, hz } in AudioContext seconds
    const recent = []; // last few raw readings, for median smoothing

    // Ends a replay and starts over from zero: the replayed audio and its pitch
    // history are discarded, so the next replay only holds what came after.
    function endReplay(node) {
      if (replayNode !== node) return;
      replayNode = null;
      ringPos = 0;
      filled = 0;
      history.length = 0;
      recent.length = 0;
    }

    function smooth(hz) {
      if (hz == null) {
        recent.length = 0;
        return null;
      }
      recent.push(hz);
      if (recent.length > 3) recent.shift();
      return median(recent);
    }

    function onAudio(event) {
      const input = event.inputBuffer.getChannelData(0);
      const t = ctx.currentTime;
      const replaying = replayNode !== null;

      // Audio and pitch history stay aligned in time. While replaying, the mic is
      // ignored (so the speakers don't get recorded) and silence is stored instead.
      for (let i = 0; i < input.length; i++) {
        ring[ringPos] = replaying ? 0 : input[i];
        ringPos = (ringPos + 1) % ring.length;
      }
      filled = Math.min(ring.length, filled + input.length);

      const raw = replaying ? null : detectPitch(input, sampleRate);
      history.push({ t, hz: smooth(raw) });

      // Keep two windows so a replay's frozen view survives while time moves on.
      const cutoff = t - 2 * WINDOW_SEC;
      while (history.length && history[0].t < cutoff) history.shift();
    }

    function releaseMic() {
      capturing = false;
      if (proc) proc.disconnect();
      if (source) source.disconnect();
      if (stream) stream.getTracks().forEach((track) => track.stop());
      proc = null;
      source = null;
      stream = null;
    }

    async function openMic() {
      if (capturing) return;
      if (replayNode) api.stopReplay();
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      if (!ctx) ctx = new (global.AudioContext || global.webkitAudioContext)();
      // Not awaited: until the page has been tapped, the browser can leave this pending,
      // and the tracker reports 'waiting' meanwhile.
      ctx.resume();
      sampleRate = ctx.sampleRate;
      ring = new Float32Array(Math.round(sampleRate * WINDOW_SEC));
      ringPos = 0;
      filled = 0;
      history.length = 0;
      recent.length = 0;

      source = ctx.createMediaStreamSource(stream);
      proc = ctx.createScriptProcessor(BLOCK, 1, 1);
      const mute = ctx.createGain();
      mute.gain.value = 0;
      proc.onaudioprocess = onAudio;
      source.connect(proc);
      proc.connect(mute);
      mute.connect(ctx.destination);
      capturing = true;
    }

    const api = {
      history,

      get state() {
        if (!ctx) return 'idle';
        if (replayNode) return 'replay';
        if (!capturing) return 'stopped';
        return ctx.state === 'running' ? 'live' : 'waiting';
      },

      // The time at the right edge of the view.
      now() {
        if (!ctx) return 0;
        if (replayNode || !capturing) return frozenNow;
        return ctx.currentTime;
      },

      // Seconds of audio available to replay.
      recordedSeconds() {
        return filled / sampleRate;
      },

      replayInfo() {
        if (!replayNode) return null;
        const elapsed = Math.min(replayDur, ctx.currentTime - replayStart);
        return { elapsed, duration: replayDur, remaining: replayDur - elapsed };
      },

      // Latest reading, or null if the last voiced moment is stale.
      current() {
        if (api.state !== 'live') return null;
        const last = history[history.length - 1];
        if (!last || last.hz == null) return null;
        if (ctx.currentTime - last.t > 0.3) return null;
        return last.hz;
      },

      pointsInView() {
        const end = api.now();
        const start = end - WINDOW_SEC;
        return history.filter((p) => p.t >= start && p.t <= end);
      },

      // Summary of the visible 10 seconds.
      stats() {
        const voiced = api.pointsInView().filter((p) => p.hz != null);
        const share = {};
        for (const band of BANDS) share[band.id] = 0;
        share.none = 0;
        for (const p of voiced) {
          const band = bandFor(p.hz);
          share[band ? band.id : 'none'] += 1;
        }
        for (const key of Object.keys(share)) {
          share[key] = voiced.length ? share[key] / voiced.length : 0;
        }
        return {
          median: median(voiced.map((p) => p.hz)),
          share,
          voicedCount: voiced.length,
        };
      },

      // Calls made while the mic is still opening share one attempt.
      start() {
        if (!starting) starting = openMic().finally(() => { starting = null; });
        return starting;
      },

      // Unpauses audio. Browsers allow this only after the page has been tapped.
      resume() {
        if (ctx) ctx.resume();
      },

      stop() {
        if (!ctx) return;
        if (replayNode) api.stopReplay();
        frozenNow = ctx.currentTime;
        releaseMic();
      },

      // Plays the last 10 seconds, skipping leading silence so playback starts just
      // before the first voiced moment. Returns false if there is nothing to play.
      replay(onEnd) {
        if (!ctx || replayNode || filled === 0) return false;
        // Each history point covers one BLOCK of samples, so counting points back from
        // the end lines them up exactly with the audio in the ring.
        const points = history.slice(-Math.ceil(filled / BLOCK));
        const firstVoiced = points.findIndex((p) => p.hz != null);
        let count = filled;
        if (firstVoiced > 0) {
          const voicedSamples = (points.length - firstVoiced) * BLOCK;
          count = Math.min(filled, voicedSamples + Math.round(PREROLL_SEC * sampleRate));
        }
        const buffer = ctx.createBuffer(1, count, sampleRate);
        const out = buffer.getChannelData(0);
        const first = (ringPos - count + ring.length) % ring.length;
        for (let i = 0; i < count; i++) out[i] = ring[(first + i) % ring.length];

        const node = ctx.createBufferSource();
        node.buffer = buffer;
        node.connect(ctx.destination);
        if (capturing) frozenNow = ctx.currentTime;
        replayStart = ctx.currentTime;
        replayDur = count / sampleRate;
        node.onended = () => {
          endReplay(node);
          if (onEnd) onEnd();
        };
        replayNode = node;
        node.start();
        return true;
      },

      stopReplay() {
        const node = replayNode;
        if (!node) return;
        endReplay(node);
        node.stop();
      },
    };
    return api;
  }

  function cssVar(el, name, fallback) {
    const value = getComputedStyle(el).getPropertyValue(name).trim();
    return value || fallback;
  }

  // Draws the scrolling pitch graph with range bands.
  // opts: { fmin, fmax, grid, labels, target (band id to highlight), dots }
  function drawGraph(canvas, tracker, opts) {
    const o = Object.assign({ fmin: 50, fmax: 450, grid: true, labels: true, target: null, dots: true }, opts);
    const dpr = global.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const g = canvas.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);

    const y = (hz) => h - ((hz - o.fmin) / (o.fmax - o.fmin)) * h;
    const font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';

    // Bands.
    for (const band of BANDS) {
      const top = y(band.hi);
      const bottom = y(band.lo);
      g.globalAlpha = o.target && o.target !== band.id ? 0.35 : 1;
      g.fillStyle = cssVar(canvas, '--band-' + band.id, '#ccc');
      g.fillRect(0, top, w, bottom - top);
      if (o.target === band.id) {
        g.globalAlpha = 1;
        g.strokeStyle = cssVar(canvas, '--target', '#2f8f5b');
        g.lineWidth = 2;
        g.strokeRect(1, top + 1, w - 2, bottom - top - 2);
      }
      if (o.labels) {
        g.globalAlpha = o.target && o.target !== band.id ? 0.5 : 1;
        g.fillStyle = cssVar(canvas, '--band-text', '#222');
        g.font = font;
        g.textAlign = 'right';
        // Thin bands get a centered label so it doesn't collide with the next band's.
        const thin = bottom - top < 24;
        g.textBaseline = thin ? 'middle' : 'top';
        g.fillText(band.label, w - 6, thin ? (top + bottom) / 2 : top + 4);
      }
    }
    g.globalAlpha = 1;

    // Grid.
    if (o.grid) {
      g.strokeStyle = cssVar(canvas, '--grid', '#999');
      g.fillStyle = cssVar(canvas, '--grid-text', '#333');
      g.lineWidth = 1;
      g.font = font;
      g.textAlign = 'left';
      g.textBaseline = 'bottom';
      for (let hz = Math.ceil(o.fmin / 50) * 50; hz <= o.fmax; hz += 50) {
        const yy = Math.round(y(hz)) + 0.5;
        g.beginPath();
        g.moveTo(0, yy);
        g.lineTo(w, yy);
        g.stroke();
        if (o.labels) g.fillText(hz + ' Hz', 6, yy - 2);
      }
    }

    // Pitch line, broken at silences.
    const end = tracker.now();
    const start = end - WINDOW_SEC;
    const x = (t) => ((t - start) / WINDOW_SEC) * w;
    const points = tracker.pointsInView();
    g.strokeStyle = cssVar(canvas, '--line', '#e0263a');
    g.lineWidth = 2;
    g.lineJoin = 'round';
    g.beginPath();
    let drawing = false;
    for (const p of points) {
      if (p.hz == null) {
        drawing = false;
        continue;
      }
      const px = x(p.t);
      const py = y(p.hz);
      if (drawing) g.lineTo(px, py);
      else g.moveTo(px, py);
      drawing = true;
    }
    g.stroke();
    if (o.dots) {
      g.fillStyle = cssVar(canvas, '--dot', '#5b2bd6');
      for (const p of points) {
        if (p.hz == null) continue;
        g.beginPath();
        g.arc(x(p.t), y(p.hz), 1.6, 0, Math.PI * 2);
        g.fill();
      }
    }

    // Replay playhead.
    const info = tracker.replayInfo();
    if (info) {
      const px = x(end - info.duration + info.elapsed);
      g.strokeStyle = cssVar(canvas, '--playhead', '#111');
      g.lineWidth = 2;
      g.beginPath();
      g.moveTo(px, 0);
      g.lineTo(px, h);
      g.stroke();
    }
  }

  // Keeps the screen on while listening, where the browser allows it. Failures are ignored.
  let wakeLock = null;
  async function keepScreenOn(on) {
    try {
      if (on && !wakeLock && navigator.wakeLock) {
        wakeLock = await navigator.wakeLock.request('screen');
        wakeLock.addEventListener('release', () => { wakeLock = null; });
      } else if (!on && wakeLock) {
        await wakeLock.release();
        wakeLock = null;
      }
    } catch (e) { /* not supported or refused */ }
  }

  // Wires a Start/Stop button and a Replay button to a tracker, and starts listening as
  // soon as the page opens. Space toggles replay. onError(message) reports mic problems.
  function bindControls(tracker, startBtn, replayBtn, onError) {
    let resumeOnReturn = false;

    async function listen() {
      try {
        await tracker.start();
        // The app was left while the mic was opening; listen again on return instead.
        if (document.visibilityState === 'hidden') {
          tracker.stop();
          resumeOnReturn = true;
          return;
        }
        keepScreenOn(true);
      } catch (err) {
        const denied = err && (err.name === 'NotAllowedError' || err.name === 'SecurityError');
        if (onError) {
          onError(denied ? 'Microphone access was blocked. Allow it in the browser and try again.' : 'Could not start the microphone: ' + (err && err.message));
        }
      }
    }

    startBtn.addEventListener('click', () => {
      if (tracker.state === 'waiting') tracker.resume();
      else if (tracker.state === 'live' || tracker.state === 'replay') {
        tracker.stop();
        keepScreenOn(false);
      } else listen();
    });
    const toggleReplay = () => {
      if (tracker.state === 'replay') tracker.stopReplay();
      else tracker.replay();
    };
    replayBtn.addEventListener('click', toggleReplay);
    // Stop listening as soon as the app is hidden (app switch, home screen, screen lock),
    // so nothing runs in the background, and listen again on return. Listening again
    // starts a fresh 10 seconds. If Stop was pressed before leaving, it stays stopped.
    document.addEventListener('visibilitychange', () => {
      const state = tracker.state;
      if (document.visibilityState === 'hidden') {
        if (state === 'live' || state === 'replay' || state === 'waiting') {
          tracker.stop();
          keepScreenOn(false);
          resumeOnReturn = true;
        }
      } else if (resumeOnReturn) {
        resumeOnReturn = false;
        listen();
      }
    });
    document.addEventListener('keydown', (e) => {
      if (e.code === 'Space' && e.target === document.body) {
        e.preventDefault();
        toggleReplay();
      }
    });

    if (document.visibilityState === 'visible') listen();
    else resumeOnReturn = true;
  }

  // Calls render every animation frame.
  function loop(render) {
    const frame = () => {
      render();
      global.requestAnimationFrame(frame);
    };
    global.requestAnimationFrame(frame);
  }

  const VoicePitch = { BANDS, WINDOW_SEC, PHRASES, bandFor, labelFor, detectPitch, median, createTracker, drawGraph, bindControls, loop };
  if (typeof module !== 'undefined' && module.exports) module.exports = VoicePitch;
  else global.VoicePitch = VoicePitch;
})(typeof window !== 'undefined' ? window : globalThis);
