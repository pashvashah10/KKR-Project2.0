/* Storefront behaviour.
 *
 * Deliberately small. The pages render server-side and mean something before
 * this file runs; everything here is enhancement, except the job poller on
 * /configure, which is genuinely asynchronous work.
 *
 * Motion follows DESIGN.md and the animate skill: only transform and opacity,
 * 200-300ms, exits ~75% of enters, and every effect checks
 * prefers-reduced-motion before it starts rather than animating invisibly.
 */

import { WeatherCanvas } from './weather-canvas.js';

const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const FINE_POINTER = matchMedia('(hover: hover) and (pointer: fine)').matches;

/* ------------------------------------------------------------------ theme */

const THEME_KEY = 'downside-theme';

function applyTheme(theme) {
  if (theme === 'system') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', theme);
}

function initTheme() {
  const stored = localStorage.getItem(THEME_KEY);
  if (stored) applyTheme(stored);

  document.querySelectorAll('[data-theme-toggle]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const dark = document.documentElement.getAttribute('data-theme') === 'dark'
        || (!document.documentElement.hasAttribute('data-theme')
            && matchMedia('(prefers-color-scheme: dark)').matches);
      const next = dark ? 'light' : 'dark';
      localStorage.setItem(THEME_KEY, next);
      applyTheme(next);
    });
  });
}

/* -------------------------------------------------------------------- nav */

function initNav() {
  const nav = document.querySelector('.nav');
  if (!nav) return;

  // Passive listener, and a single attribute flip rather than style writes.
  let ticking = false;
  const onScroll = () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      nav.toggleAttribute('data-scrolled', scrollY > 40);
      ticking = false;
    });
  };
  addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  const toggle = nav.querySelector('[data-nav-toggle]');
  if (toggle) {
    toggle.addEventListener('click', () => {
      const open = nav.toggleAttribute('data-open');
      toggle.setAttribute('aria-expanded', String(open));
    });
  }
}

/* ---------------------------------------------------------------- reveals */

function initReveals() {
  const items = document.querySelectorAll('.reveal');
  if (!items.length) return;
  if (REDUCED) {
    items.forEach((el) => el.setAttribute('data-shown', ''));
    return;
  }

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.setAttribute('data-shown', '');
        io.unobserve(entry.target);
      });
    },
    { rootMargin: '0px 0px -12% 0px', threshold: 0.06 },
  );
  items.forEach((el) => io.observe(el));
}

/* -------------------------------------------------------------- spotlight */

function initSpotlight() {
  if (!FINE_POINTER || REDUCED) return;
  const cards = document.querySelectorAll('.card');
  if (!cards.length) return;

  let frame = null;
  let pending = null;

  const flush = () => {
    frame = null;
    if (!pending) return;
    const { card, x, y } = pending;
    card.style.setProperty('--mx', `${x}px`);
    card.style.setProperty('--my', `${y}px`);
    pending = null;
  };

  cards.forEach((card) => {
    card.addEventListener('pointermove', (event) => {
      const rect = card.getBoundingClientRect();
      pending = { card, x: event.clientX - rect.left, y: event.clientY - rect.top };
      if (frame === null) frame = requestAnimationFrame(flush);
    }, { passive: true });
  });
}

/* --------------------------------------------------------------- magnetic */

function initMagnetic() {
  if (!FINE_POINTER || REDUCED) return;
  const targets = document.querySelectorAll('[data-magnetic]');
  const MAX = 5;

  targets.forEach((el) => {
    let frame = null;
    let next = null;

    const apply = () => {
      frame = null;
      if (!next) return;
      el.style.transform = `translate(${next.x}px, ${next.y}px)`;
      next = null;
    };

    el.addEventListener('pointermove', (event) => {
      const r = el.getBoundingClientRect();
      const dx = (event.clientX - (r.left + r.width / 2)) / (r.width / 2);
      const dy = (event.clientY - (r.top + r.height / 2)) / (r.height / 2);
      next = {
        x: Math.max(-1, Math.min(1, dx)) * MAX,
        y: Math.max(-1, Math.min(1, dy)) * MAX,
      };
      if (frame === null) frame = requestAnimationFrame(apply);
    }, { passive: true });

    el.addEventListener('pointerleave', () => {
      if (frame !== null) { cancelAnimationFrame(frame); frame = null; }
      el.style.transform = '';
    });
  });
}

/* ------------------------------------------------------------- count-up */

function initCounters() {
  const nodes = document.querySelectorAll('[data-count]');
  if (!nodes.length) return;
  if (REDUCED) {
    nodes.forEach((n) => { n.textContent = n.dataset.count; });
    return;
  }

  const easeOutCubic = (t) => 1 - (1 - t) ** 3;

  const run = (node) => {
    const text = node.dataset.count;
    const target = parseFloat(text.replace(/[^0-9.-]/g, ''));
    if (!Number.isFinite(target)) { node.textContent = text; return; }
    const decimals = (text.split('.')[1] || '').replace(/[^0-9]/g, '').length;
    const prefix = text.match(/^[^0-9.-]*/)[0];
    const suffix = text.match(/[^0-9.,]*$/)[0];
    // Group only if the source did. A year is not "2,125".
    const grouped = text.includes(',');
    const started = performance.now();
    const DURATION = 900;

    const step = (now) => {
      const t = Math.min((now - started) / DURATION, 1);
      const value = target * easeOutCubic(t);
      node.textContent = prefix
        + value.toLocaleString(undefined, {
            useGrouping: grouped,
            minimumFractionDigits: decimals, maximumFractionDigits: decimals })
        + suffix;
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };

  const io = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      run(entry.target);
      io.unobserve(entry.target);
    });
  }, { threshold: 0.4 });
  nodes.forEach((n) => io.observe(n));
}

/* ------------------------------------------------------------ the canvas */

/* Named presets rather than raw parameters, so a template can ask for
 * `weather="rain"` without knowing the simulator's field names. Each is a
 * plausible season rather than a caricature: `rain` is a wet maritime autumn,
 * `snow` a continental winter. */
const SCENES = {
  clear: { wet: 0.10, heavy: 0.02, snow: 0.00, windy: 0.10, hot: 0.34, tmax: 24, wind: 3.4 },
  cloud: { wet: 0.34, heavy: 0.06, snow: 0.00, windy: 0.26, hot: 0.10, tmax: 16, wind: 5.0 },
  rain:  { wet: 0.78, heavy: 0.34, snow: 0.00, windy: 0.46, hot: 0.03, tmax: 12, wind: 7.2 },
  snow:  { wet: 0.62, heavy: 0.12, snow: 0.82, windy: 0.38, hot: 0.00, tmax: -3, wind: 5.8 },
};

function initCanvas() {
  const canvas = document.getElementById('sky');
  if (!canvas || REDUCED) return null;

  const sky = new WeatherCanvas(canvas, { intensity: 0.9 });
  const scene = document.body.dataset.weather || 'clear';
  sky.set(SCENES[scene] || SCENES.clear);
  sky.start();

  // Stop the loop when the tab is hidden. A background rAF is pure waste.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) sky.stop();
    else sky.start();
  });

  return sky;
}

/* --------------------------------------------------------- job polling */

/* The configure page's async fit.
 *
 * POST returns immediately with a job id; this polls every 3s and reports the
 * staged progress that `service.fit_site` writes as it goes. The connection is
 * never held open for the 80-140 seconds a century fit takes.
 */

const POLL_MS = 3000;

function setStep(root, text, fraction) {
  const label = root.querySelector('[data-step]');
  const bar = root.querySelector('[data-progress]');
  if (label && text) label.textContent = text;
  if (bar && typeof fraction === 'number') bar.style.setProperty('--p', String(fraction));
}

async function pollJob(root, jobId, onDone, onFail) {
  let stopped = false;

  const tick = async () => {
    if (stopped) return;
    let data;
    try {
      const res = await fetch(`/api/v1/jobs/${jobId}`, { headers: { Accept: 'application/json' } });
      if (!res.ok) throw new Error(`status ${res.status}`);
      data = await res.json();
    } catch (err) {
      // A single failed poll is not a failed fit --- the worker is still going.
      // Keep trying; only an explicit `failed` status ends this.
      setTimeout(tick, POLL_MS);
      return;
    }

    setStep(root, data.step, data.progress);

    if (data.status === 'succeeded') { stopped = true; onDone(data); return; }
    if (data.status === 'failed') { stopped = true; onFail(data); return; }
    setTimeout(tick, POLL_MS);
  };

  tick();
  return () => { stopped = true; };
}

function initConfigure() {
  const root = document.querySelector('[data-configure]');
  if (!root) return;

  const form = root.querySelector('[data-venue-form]');
  const progress = root.querySelector('[data-fit]');
  const errorBox = root.querySelector('[data-fit-error]');
  const stationBox = root.querySelector('[data-station-result]');
  const slug = root.dataset.slug;

  const show = (el) => el && el.removeAttribute('hidden');
  const hide = (el) => el && el.setAttribute('hidden', '');

  // Captured from the POST so the "come back later" link is on screen while the
  // fit runs, not only after it lands.
  let resumeUrl = null;

  const finish = (data) => {
    // Re-render server-side rather than assembling the quote in JavaScript:
    // the price must come from the engine, and the page already knows how to
    // display it. Prefer the signed link so the resulting URL is shareable.
    location.href = data.magic_link || resumeUrl || `/configure/${slug}?site=${data.site_id}`;
  };

  const fail = (data) => {
    hide(progress);
    if (errorBox) {
      errorBox.textContent = data.error
        ? `Fitting failed: ${data.error}`
        : 'Fitting failed. Please try again, or pick an example venue.';
      show(errorBox);
    }
  };

  // Resume a job the visitor came back to via a link.
  const resume = root.dataset.resumeJob;
  if (resume) {
    show(progress);
    pollJob(root, resume, finish, fail);
  }

  if (!form) return;

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    hide(errorBox);

    const submit = form.querySelector('[type="submit"]');
    if (submit) { submit.disabled = true; submit.textContent = 'Starting…'; }

    let data;
    try {
      const res = await fetch(`/configure/${slug}/venue`, {
        method: 'POST',
        body: new FormData(form),
      });
      data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'could not start the fit');
    } catch (err) {
      if (submit) { submit.disabled = false; submit.textContent = 'Match station and fit'; }
      if (errorBox) { errorBox.textContent = String(err.message || err); show(errorBox); }
      return;
    }

    // The station match is synchronous and comes back in this response --- it
    // is the first thing worth seeing, and it arrives long before the fit.
    if (stationBox && data.station) {
      stationBox.innerHTML = renderStation(data.station);
      show(stationBox);
    }

    resumeUrl = data.resume_link || null;
    const resumeBox = root.querySelector('[data-resume]');
    if (resumeBox && resumeUrl) {
      // Rendered now rather than on completion: the whole point is that the
      // visitor can leave, and they cannot leave with a link they have not
      // been given yet.
      resumeBox.innerHTML = `
        <p class="eyebrow">Leaving? Come back to this</p>
        <p class="small">${
          data.notify && !data.mail_configured
            ? 'Email delivery is not configured on this deployment, so nothing will be sent. Bookmark this — it reopens this venue from any device.'
            : data.notify
              ? 'We will email this link when the fit lands. It also works right now:'
              : 'Bookmark this — it reopens this venue from any device, with no sign-in.'
        }</p>
        <p class="num small" style="margin-top:var(--sp-3);word-break:break-all;
                  background:var(--panel-2);padding:var(--sp-3);border-radius:3px">
          <a href="${esc(resumeUrl)}">${esc(resumeUrl)}</a>
        </p>`;
      show(resumeBox);
    }

    form.setAttribute('hidden', '');
    show(progress);
    setStep(root, 'Queued', 0.02);
    pollJob(root, data.job_id, finish, fail);
  });
}

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function renderStation(station) {
  if (!station || !station.matched) {
    return `<div class="notice"><span>${esc(station ? station.note : 'No station matched.')}</span></div>`;
  }
  const warn = station.elevation_warning && station.note
    ? `<div class="notice notice-crit" style="margin-top:var(--sp-3)"><span>${esc(station.note)}</span></div>`
    : '';
  return `
    <div class="panel">
      <p class="eyebrow">Settles on</p>
      <p class="h-card">${esc(station.name)} <span class="num dim">${esc(station.id)}</span></p>
      <dl class="readout" style="margin-top:var(--sp-3)">
        <dt>Distance</dt><dd class="num">${esc(station.distance_km)} km</dd>
        <dt>Record</dt><dd class="num">${esc(station.first_year)}–${esc(station.last_year)}</dd>
        <dt>Station elevation</dt><dd class="num">${esc(Math.round(station.elevation_m))} m</dd>
      </dl>
    </div>${warn}`;
}

/* ------------------------------------------------------------------ boot */

initTheme();
initNav();
initReveals();
initSpotlight();
initMagnetic();
initCounters();
initCanvas();
initConfigure();
