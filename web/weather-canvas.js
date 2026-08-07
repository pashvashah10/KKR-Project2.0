/* Ambient weather.
 *
 * Not a decorative loop. Every parameter comes from the same simulation that
 * produces the charts --- wet fraction, heavy-rain fraction, snow fraction, windy
 * fraction, hot fraction and mean daily maximum for the selected site, year and
 * pathway. Move the year to 2125 under a high-emissions path and the rain thins
 * and the light hardens because the model says it does.
 *
 * What makes it read as weather rather than as falling dots:
 *
 *   depth      three parallax planes. Far particles are small, slow, pale and
 *              slightly blurred; near ones are large, fast and sharp. A single
 *              plane always looks like a screensaver.
 *   gusting    wind is smooth value noise, not a constant. Real wind surges and
 *              drops, and every particle responds to the same field, so the
 *              whole scene breathes together.
 *   light      the sun is volumetric --- a soft disc plus long rays that rotate
 *              slowly, occluded by drifting cloud. Overcast suppresses it.
 *   inertia    state changes ease over about two seconds, so switching pathway
 *              reads as a climate drifting rather than a slide transition.
 */

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = (t) => t * t * (3 - 2 * t);

/** Smooth 1-D value noise. Cheap, and enough for a wind field. */
function makeNoise(seed = 1) {
  const table = new Float32Array(512);
  let s = seed;
  for (let i = 0; i < 512; i++) {
    s = (s * 1664525 + 1013904223) % 4294967296;
    table[i] = s / 4294967296;
  }
  return (x) => {
    const i = Math.floor(x);
    const f = smooth(x - i);
    return lerp(table[i & 511], table[(i + 1) & 511], f);
  };
}

const DEFAULT = { wet: 0, heavy: 0, snow: 0, windy: 0, hot: 0, tmax: 15, wind: 4 };

const LAYERS = [
  // depth,  speed, size, alpha
  { z: 0.35, v: 0.45, s: 0.55, a: 0.34 },
  { z: 0.65, v: 0.75, s: 0.8, a: 0.62 },
  { z: 1.0, v: 1.15, s: 1.15, a: 1.0 },
];

export class WeatherCanvas {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: true });
    this.state = { ...DEFAULT };
    this.target = { ...DEFAULT };
    this.reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.intensity = opts.intensity ?? 1;
    this.running = false;
    this.t = 0;
    this.flash = 0;
    this.nextFlash = 5000;
    this.sunAngle = 0;

    this.gustNoise = makeNoise(7);
    this.cloudNoise = makeNoise(23);

    this.rain = [];
    this.snow = [];
    this.gust = [];
    this.splash = [];
    this.MAX_RAIN = 420;
    this.MAX_SNOW = 320;
    this.MAX_GUST = 46;

    this._resize = this._resize.bind(this);
    this._frame = this._frame.bind(this);
    this._resize();
    addEventListener('resize', this._resize, { passive: true });
  }

  _resize() {
    const r = this.canvas.getBoundingClientRect();
    // Cap the backing store. This sits behind the whole app; a 3x buffer on a
    // large display costs far more than the effect is worth.
    const dpr = Math.min(devicePixelRatio || 1, 2);
    this.w = Math.max(r.width, 1);
    this.h = Math.max(r.height, 1);
    this.canvas.width = Math.round(this.w * dpr);
    this.canvas.height = Math.round(this.h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  set(next) {
    Object.assign(this.target, next);
    if (this.reduced) Object.assign(this.state, next);
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.last = performance.now();
    requestAnimationFrame(this._frame);
  }
  stop() { this.running = false; }

  /* --- particles ------------------------------------------------------- */
  _newRain(layer) {
    const L = LAYERS[layer];
    return {
      L: layer,
      x: Math.random() * (this.w + 400) - 200,
      y: Math.random() * -this.h,
      len: (10 + Math.random() * 22) * L.s,
      v: (620 + Math.random() * 520) * L.v,
      a: (0.14 + Math.random() * 0.3) * L.a,
    };
  }
  _newSnow(layer) {
    const L = LAYERS[layer];
    return {
      L: layer,
      x: Math.random() * this.w,
      y: Math.random() * -this.h,
      r: (1.0 + Math.random() * 2.4) * L.s,
      v: (26 + Math.random() * 44) * L.v,
      phase: Math.random() * 6.283,
      sway: 0.5 + Math.random() * 1.3,
      a: (0.28 + Math.random() * 0.5) * L.a,
    };
  }
  _newGust() {
    return {
      x: Math.random() * this.w,
      y: Math.random() * this.h,
      len: 60 + Math.random() * 200,
      v: 200 + Math.random() * 420,
      a: 0.04 + Math.random() * 0.11,
    };
  }

  _frame(now) {
    if (!this.running) return;
    const dt = Math.min((now - this.last) / 1000, 0.05);
    this.last = now;
    this.t += dt;

    // Ease toward the target: a scenario change should drift, not cut.
    const k = 1 - Math.pow(0.0025, dt);
    for (const key in this.target) {
      this.state[key] = lerp(this.state[key], this.target[key], k);
    }
    this._draw(dt);
    requestAnimationFrame(this._frame);
  }

  _draw(dt) {
    const { ctx, w, h, state } = this;
    ctx.clearRect(0, 0, w, h);

    // Wind field: a base from the data plus gusting, so the scene surges.
    const gust = this.gustNoise(this.t * 0.22) * 0.7 + this.gustNoise(this.t * 0.9) * 0.3;
    const windBase = clamp(state.wind / 13, 0, 1);
    const wind = windBase * (0.55 + 0.9 * gust) + state.windy * 0.35 * gust;
    const slant = clamp(lerp(0.05, 0.85, wind), 0, 1.1);

    this._sky();
    this._clouds();
    this._sun(dt);

    const g = this.intensity;

    /* rain --------------------------------------------------------------- */
    const wantRain = Math.round(
      this.MAX_RAIN * clamp(state.wet * (0.45 + state.heavy * 2.6), 0, 1) * g
    );
    this._top(this.rain, wantRain, (i) => this._newRain(i % 3));
    if (this.rain.length) {
      ctx.lineCap = 'round';
      for (let li = 0; li < 3; li++) {
        ctx.beginPath();
        ctx.strokeStyle = this.rainColor;
        ctx.lineWidth = 0.7 + li * 0.45;
        for (const p of this.rain) {
          if (p.L !== li) continue;
          if (!this.reduced) {
            p.y += p.v * dt;
            p.x += p.v * dt * slant * LAYERS[li].v * 0.55;
          }
          if (p.y > h + 24 || p.x > w + 220) {
            Object.assign(p, this._newRain(li));
            p.y = -24;
            if (state.heavy > 0.02 && Math.random() < 0.05) this._addSplash(p.x, h);
          }
          ctx.globalAlpha = p.a;
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(p.x - p.len * slant, p.y - p.len);
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    this._drawSplash(dt);

    /* snow --------------------------------------------------------------- */
    const wantSnow = Math.round(this.MAX_SNOW * clamp(state.snow * 2.8, 0, 1) * g);
    this._top(this.snow, wantSnow, (i) => this._newSnow(i % 3));
    if (this.snow.length) {
      ctx.fillStyle = this.snowColor;
      for (const p of this.snow) {
        if (!this.reduced) {
          p.y += p.v * dt;
          p.phase += dt * p.sway;
          // Turbulence: swirl plus the shared gust, so flakes eddy together.
          p.x += (Math.sin(p.phase) * 16 + Math.cos(p.phase * 0.4) * 7 + wind * 120) * dt;
        }
        if (p.y > h + 10) { Object.assign(p, this._newSnow(p.L)); p.y = -10; }
        if (p.x > w + 12) p.x = -12;
        if (p.x < -12) p.x = w + 12;
        ctx.globalAlpha = p.a;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, 6.2832);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    /* wind streaks -------------------------------------------------------- */
    const wantGust = Math.round(this.MAX_GUST * clamp(state.windy * 2.0, 0, 1) * g * (0.4 + gust));
    this._top(this.gust, wantGust, () => this._newGust());
    if (this.gust.length) {
      ctx.strokeStyle = this.gustColor;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (const p of this.gust) {
        if (!this.reduced) p.x += p.v * dt * (0.5 + gust);
        if (p.x - p.len > w) { Object.assign(p, this._newGust()); p.x = -p.len; }
        ctx.globalAlpha = p.a * (0.4 + gust * 0.8);
        const yy = p.y + Math.sin(this.t * 0.8 + p.y * 0.02) * 7;
        ctx.moveTo(p.x - p.len, yy);
        ctx.lineTo(p.x, yy);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    this._lightning(dt);
  }

  _top(arr, want, make) {
    while (arr.length < want) arr.push(make(arr.length));
    if (arr.length > want) arr.length = want;
  }

  _addSplash(x, y) {
    if (this.splash.length > 60) return;
    this.splash.push({ x, y: y - 2, r: 0, life: 1 });
  }
  _drawSplash(dt) {
    if (!this.splash.length || this.reduced) return;
    const { ctx } = this;
    ctx.strokeStyle = this.rainColor;
    ctx.lineWidth = 1;
    for (let i = this.splash.length - 1; i >= 0; i--) {
      const s = this.splash[i];
      s.life -= dt * 1.9;
      s.r += dt * 42;
      if (s.life <= 0) { this.splash.splice(i, 1); continue; }
      ctx.globalAlpha = s.life * 0.28;
      ctx.beginPath();
      ctx.ellipse(s.x, s.y, s.r, s.r * 0.32, 0, 0, 6.2832);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  /* --- atmosphere ------------------------------------------------------- */
  _sky() {
    const { ctx, w, h, state } = this;
    const warm = clamp((state.tmax - 2) / 36, 0, 1);
    const grd = ctx.createLinearGradient(0, 0, 0, h);
    grd.addColorStop(0, this.skyTop(warm));
    grd.addColorStop(0.55, this.skyMid(warm));
    grd.addColorStop(1, this.skyBottom(warm));
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, w, h);
  }

  _clouds() {
    const { ctx, w, h, state } = this;
    const cover = clamp(state.wet * 1.7 + state.heavy * 0.8, 0, 1);
    if (cover < 0.04) return;
    // Three slow, soft masses at different speeds. Drawn as radial gradients
    // rather than sprites so they stay smooth at any size.
    for (let i = 0; i < 3; i++) {
      const drift = this.cloudNoise(this.t * 0.02 + i * 40);
      const cx = ((this.t * (7 + i * 5) + drift * 900 + i * 520) % (w + 900)) - 450;
      const cy = h * (0.10 + i * 0.13) + Math.sin(this.t * 0.05 + i) * 18;
      const rad = Math.min(w, h) * (0.45 + i * 0.16);
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, rad);
      g.addColorStop(0, this.cloudColor(cover * (0.5 + 0.5 * (i / 2))));
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, h);
    }
  }

  _sun(dt) {
    const { ctx, w, h, state } = this;
    const heat = clamp(state.hot, 0, 1);
    const clear = clamp(1 - state.wet * 1.6, 0, 1);
    const power = heat * 0.68 + clear * 0.32;
    if (power < 0.04) return;

    if (!this.reduced) this.sunAngle += dt * 0.035;
    const cx = w * 0.79;
    const cy = h * 0.2;
    const rad = Math.min(w, h) * (0.4 + heat * 0.34);

    // Volumetric rays. Long, soft wedges that turn slowly --- this is what makes
    // strong light read as glare rather than as a gradient blob.
    if (power > 0.18) {
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(this.sunAngle);
      const rays = 11;
      for (let i = 0; i < rays; i++) {
        const a0 = (i / rays) * 6.2832;
        const wob = 0.05 + 0.03 * Math.sin(this.t * 0.7 + i);
        const g = ctx.createLinearGradient(0, 0, Math.cos(a0) * rad * 2.1, Math.sin(a0) * rad * 2.1);
        g.addColorStop(0, this.sunCore(power * 0.5));
        g.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.arc(0, 0, rad * 2.1, a0 - wob, a0 + wob);
        ctx.closePath();
        ctx.fill();
      }
      ctx.restore();
    }

    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, rad);
    g.addColorStop(0, this.sunCore(power));
    g.addColorStop(0.4, this.sunMid(power));
    g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);

    // Heat shimmer, only when genuinely hot, so it reads as a signal.
    if (heat > 0.32 && !this.reduced) {
      ctx.save();
      ctx.globalAlpha = (heat - 0.32) * 0.2;
      ctx.strokeStyle = this.sunCore(1);
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < 8; i++) {
        const yy = h * (0.52 + i * 0.06);
        ctx.moveTo(0, yy);
        for (let x = 0; x <= w; x += 22) {
          ctx.lineTo(x, yy + Math.sin(x * 0.021 + this.t * 1.7 + i) * (2.4 + i * 0.3));
        }
      }
      ctx.stroke();
      ctx.restore();
    }
  }

  _lightning(dt) {
    const { ctx, w, h, state } = this;
    if (this.reduced) return;
    const rate = clamp(state.heavy * 2.4, 0, 1);
    if (rate > 0.02) {
      this.nextFlash -= dt * 1000 * rate;
      if (this.nextFlash <= 0) {
        this.flash = 1;
        this.nextFlash = 3200 + Math.random() * 9000;
      }
    }
    if (this.flash > 0) {
      this.flash = Math.max(0, this.flash - dt * 3.1);
      // Double-strobe: a real strike flickers.
      const f = this.flash > 0.72 ? this.flash * 0.55 : this.flash;
      ctx.fillStyle = `rgba(198,220,255,${Math.pow(f, 3) * 0.34})`;
      ctx.fillRect(0, 0, w, h);
    }
  }

  /* --- palette ---------------------------------------------------------- */
  get dark() {
    const stamp = document.documentElement.getAttribute('data-theme');
    if (stamp === 'dark') return true;
    if (stamp === 'light') return false;
    return matchMedia('(prefers-color-scheme: dark)').matches;
  }

  get rainColor() { return this.dark ? 'rgba(158,198,228,0.9)' : 'rgba(40,84,120,0.8)'; }
  get snowColor() { return this.dark ? 'rgba(230,243,255,0.95)' : 'rgba(96,138,172,0.95)'; }
  get gustColor() { return this.dark ? 'rgba(182,208,230,0.5)' : 'rgba(58,100,138,0.55)'; }

  cloudColor(a) {
    return this.dark ? `rgba(38,58,72,${0.30 * a})` : `rgba(104,134,156,${0.26 * a})`;
  }
  skyTop(warm) {
    return this.dark
      ? `rgba(${Math.round(lerp(14, 58, warm))},${Math.round(lerp(32, 34, warm))},${Math.round(lerp(46, 30, warm))},0.9)`
      : `rgba(${Math.round(lerp(198, 246, warm))},${Math.round(lerp(220, 226, warm))},${Math.round(lerp(233, 202, warm))},0.9)`;
  }
  skyMid(warm) {
    return this.dark
      ? `rgba(${Math.round(lerp(11, 40, warm))},${Math.round(lerp(23, 24, warm))},${Math.round(lerp(31, 22, warm))},0.55)`
      : `rgba(${Math.round(lerp(222, 250, warm))},${Math.round(lerp(233, 236, warm))},${Math.round(lerp(241, 220, warm))},0.55)`;
  }
  skyBottom(warm) {
    return this.dark
      ? `rgba(${Math.round(lerp(8, 24, warm))},${Math.round(lerp(15, 14, warm))},${Math.round(lerp(20, 15, warm))},0.35)`
      : `rgba(${Math.round(lerp(238, 252, warm))},${Math.round(lerp(243, 244, warm))},${Math.round(lerp(246, 234, warm))},0.4)`;
  }
  sunCore(p) { return this.dark ? `rgba(255,190,116,${0.19 * p})` : `rgba(255,201,128,${0.26 * p})`; }
  sunMid(p) { return this.dark ? `rgba(226,146,70,${0.07 * p})` : `rgba(236,166,92,${0.11 * p})`; }
}
