/* Ambient weather renderer.
 *
 * This is not a decorative loop. Every parameter it draws comes from the same
 * simulation that produces the charts: `ambient[scenario][year]` carries wet
 * fraction, heavy-rain fraction, snow fraction, windy fraction, hot fraction and
 * mean temperature for the selected site, year and pathway.
 *
 * So moving the year scrubber to 2125 under SSP5-8.5 thins the rain and hardens
 * the light because the model says it does. The backdrop and the numbers are the
 * same finding rendered twice.
 *
 * State changes are eased rather than cut, so a scenario switch reads as a
 * climate drifting rather than a slide transition.
 */

const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

const TARGET = {
  wet: 0, heavy: 0, snow: 0, windy: 0, hot: 0, tmax: 15, wind: 4,
};

export class WeatherCanvas {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: true });
    this.state = { ...TARGET };
    this.target = { ...TARGET };
    this.reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.intensity = opts.intensity ?? 1;
    this.running = false;
    this.t = 0;
    this.flash = 0;
    this.nextFlash = 400;

    this.rain = [];
    this.snow = [];
    this.gust = [];
    this.MAX_RAIN = 340;
    this.MAX_SNOW = 260;
    this.MAX_GUST = 40;

    this._resize = this._resize.bind(this);
    this._frame = this._frame.bind(this);
    this._resize();
    addEventListener('resize', this._resize, { passive: true });
  }

  _resize() {
    const r = this.canvas.getBoundingClientRect();
    // Cap the backing store: this layer sits behind the whole app and a 3x
    // buffer on a large monitor costs more than the effect is worth.
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

  _spawnRain() {
    return {
      x: Math.random() * (this.w + 260) - 130,
      y: Math.random() * -this.h,
      len: 8 + Math.random() * 16,
      v: 480 + Math.random() * 520,
      a: 0.16 + Math.random() * 0.34,
    };
  }

  _spawnSnow() {
    return {
      x: Math.random() * this.w,
      y: Math.random() * -this.h,
      r: 0.9 + Math.random() * 2.1,
      v: 22 + Math.random() * 46,
      drift: Math.random() * Math.PI * 2,
      sway: 0.4 + Math.random() * 1.1,
      a: 0.25 + Math.random() * 0.5,
    };
  }

  _spawnGust() {
    return {
      x: Math.random() * this.w,
      y: Math.random() * this.h,
      len: 40 + Math.random() * 130,
      v: 150 + Math.random() * 320,
      a: 0.05 + Math.random() * 0.13,
    };
  }

  _frame(now) {
    if (!this.running) return;
    const dt = Math.min((now - this.last) / 1000, 0.05);
    this.last = now;
    this.t += dt;

    // Ease the visible state toward the target so a scenario change reads as a
    // climate drifting rather than a hard cut.
    const k = 1 - Math.pow(0.0016, dt);
    for (const key in this.target) {
      this.state[key] = lerp(this.state[key], this.target[key], k);
    }

    this._draw(dt);
    requestAnimationFrame(this._frame);
  }

  _draw(dt) {
    const { ctx, w, h, state } = this;
    ctx.clearRect(0, 0, w, h);

    const wind = clamp(state.wind / 12, 0, 1);
    const slant = lerp(0.06, 0.62, wind) * (1 + state.windy * 0.8);

    this._drawSky();
    this._drawSun();

    const g = this.intensity;

    // --- rain -------------------------------------------------------------
    const wantRain = Math.round(this.MAX_RAIN * clamp(state.wet * (0.5 + state.heavy * 2.4), 0, 1) * g);
    while (this.rain.length < wantRain) this.rain.push(this._spawnRain());
    if (this.rain.length > wantRain) this.rain.length = wantRain;

    if (this.rain.length) {
      ctx.lineCap = 'round';
      ctx.strokeStyle = this.rainColor;
      ctx.lineWidth = 1.05;
      ctx.beginPath();
      for (const p of this.rain) {
        if (!this.reduced) {
          p.y += p.v * dt;
          p.x += p.v * dt * slant;
        }
        if (p.y > h + 20 || p.x > w + 140) { Object.assign(p, this._spawnRain()); p.y = -20; }
        ctx.globalAlpha = p.a;
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x - p.len * slant, p.y - p.len);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // --- snow -------------------------------------------------------------
    const wantSnow = Math.round(this.MAX_SNOW * clamp(state.snow * 2.6, 0, 1) * g);
    while (this.snow.length < wantSnow) this.snow.push(this._spawnSnow());
    if (this.snow.length > wantSnow) this.snow.length = wantSnow;

    if (this.snow.length) {
      ctx.fillStyle = this.snowColor;
      for (const p of this.snow) {
        if (!this.reduced) {
          p.y += p.v * dt;
          p.drift += dt * p.sway;
          p.x += (Math.sin(p.drift) * 14 + wind * 90) * dt;
        }
        if (p.y > h + 8) { Object.assign(p, this._spawnSnow()); p.y = -8; }
        if (p.x > w + 8) p.x = -8;
        if (p.x < -8) p.x = w + 8;
        ctx.globalAlpha = p.a;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, 6.2832);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    // --- wind streaks -------------------------------------------------------
    const wantGust = Math.round(this.MAX_GUST * clamp(state.windy * 2.2, 0, 1) * g);
    while (this.gust.length < wantGust) this.gust.push(this._spawnGust());
    if (this.gust.length > wantGust) this.gust.length = wantGust;

    if (this.gust.length) {
      ctx.strokeStyle = this.gustColor;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      for (const p of this.gust) {
        if (!this.reduced) p.x += p.v * dt;
        if (p.x - p.len > w) { Object.assign(p, this._spawnGust()); p.x = -p.len; }
        ctx.globalAlpha = p.a;
        const yy = p.y + Math.sin(this.t * 0.7 + p.y) * 5;
        ctx.moveTo(p.x - p.len, yy);
        ctx.lineTo(p.x, yy);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    this._drawLightning(dt);
  }

  _drawSky() {
    const { ctx, w, h, state } = this;
    // Warmth of the ground follows mean daily maximum.
    const warm = clamp((state.tmax - 4) / 34, 0, 1);
    const grd = ctx.createLinearGradient(0, 0, 0, h);
    grd.addColorStop(0, this.skyTop(warm));
    grd.addColorStop(1, this.skyBottom(warm));
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, w, h);
  }

  _drawSun() {
    const { ctx, w, h, state } = this;
    const heat = clamp(state.hot, 0, 1);
    // Overcast suppresses the disc; heat sharpens it.
    const clear = clamp(1 - state.wet * 1.5, 0, 1);
    const power = heat * 0.75 + clear * 0.25;
    if (power < 0.03) return;

    const cx = w * 0.78;
    const cy = h * 0.22;
    const pulse = this.reduced ? 0 : Math.sin(this.t * 0.5) * 0.04;
    const rad = Math.min(w, h) * (0.42 + heat * 0.3 + pulse);

    const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, rad);
    grd.addColorStop(0, this.sunCore(power));
    grd.addColorStop(0.45, this.sunMid(power));
    grd.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, w, h);

    // Heat shimmer: only when it is genuinely hot, so it reads as a signal.
    if (heat > 0.35 && !this.reduced) {
      ctx.save();
      ctx.globalAlpha = (heat - 0.35) * 0.16;
      ctx.strokeStyle = this.sunCore(1);
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < 7; i++) {
        const yy = h * (0.55 + i * 0.06);
        ctx.moveTo(0, yy);
        for (let x = 0; x <= w; x += 26) {
          ctx.lineTo(x, yy + Math.sin(x * 0.02 + this.t * 1.6 + i) * 3.2);
        }
      }
      ctx.stroke();
      ctx.restore();
    }
  }

  _drawLightning(dt) {
    const { ctx, w, h, state } = this;
    if (this.reduced) return;
    // Strike rate scales with heavy-rain fraction; nothing fires in a dry year.
    const rate = clamp(state.heavy * 2.2, 0, 1);
    if (rate > 0.02) {
      this.nextFlash -= dt * 1000 * rate;
      if (this.nextFlash <= 0) {
        this.flash = 1;
        this.nextFlash = 2600 + Math.random() * 7000;
      }
    }
    if (this.flash > 0) {
      this.flash = Math.max(0, this.flash - dt * 3.4);
      const a = Math.pow(this.flash, 3) * 0.4;
      ctx.fillStyle = `rgba(190,215,255,${a})`;
      ctx.fillRect(0, 0, w, h);
    }
  }

  /* --- palette hooks, overridden per theme -------------------------------- */
  get rainColor() { return this.dark ? 'rgba(150,190,225,0.85)' : 'rgba(70,110,150,0.6)'; }
  get snowColor() { return this.dark ? 'rgba(226,240,252,0.95)' : 'rgba(150,180,210,0.8)'; }
  get gustColor() { return this.dark ? 'rgba(180,205,230,0.5)' : 'rgba(90,130,170,0.4)'; }

  skyTop(warm) {
    return this.dark
      ? `rgba(${Math.round(lerp(14, 52, warm))},${Math.round(lerp(26, 30, warm))},${Math.round(lerp(42, 30, warm))},0.85)`
      : `rgba(${Math.round(lerp(206, 246, warm))},${Math.round(lerp(224, 226, warm))},${Math.round(lerp(240, 205, warm))},0.85)`;
  }
  skyBottom(warm) {
    return this.dark
      ? `rgba(${Math.round(lerp(8, 26, warm))},${Math.round(lerp(13, 15, warm))},${Math.round(lerp(18, 16, warm))},0.4)`
      : `rgba(${Math.round(lerp(238, 252, warm))},${Math.round(lerp(243, 242, warm))},${Math.round(lerp(248, 232, warm))},0.5)`;
  }
  sunCore(p) {
    return this.dark ? `rgba(255,186,110,${0.20 * p})` : `rgba(255,198,120,${0.30 * p})`;
  }
  sunMid(p) {
    return this.dark ? `rgba(232,151,74,${0.07 * p})` : `rgba(240,170,90,${0.12 * p})`;
  }

  get dark() {
    const stamp = document.documentElement.getAttribute('data-theme');
    if (stamp === 'dark') return true;
    if (stamp === 'light') return false;
    return matchMedia('(prefers-color-scheme: dark)').matches;
  }
}
