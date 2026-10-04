// Structural Affinity playground: 2D CBD vs CBD-SA with live mean-value coordinates.
// Notation follows the paper: inner cage V (N handles), outer cage U (M handles),
// W[i][k] = MVC of V_i w.r.t. U, C[i][j] = sum_k min(W[i][k], W[j][k]),
// dV = D + alpha * (C - I)^T D.
(function () {
  'use strict';

  // ---------- geometry helpers ----------
  function mvc(p, poly) {
    // Floater 2D mean-value coordinates of point p w.r.t. closed polygon poly.
    var n = poly.length, w = new Float64Array(n), r = new Float64Array(n), t = new Float64Array(n);
    for (var i = 0; i < n; i++) {
      var dx = poly[i][0] - p[0], dy = poly[i][1] - p[1];
      r[i] = Math.hypot(dx, dy);
      if (r[i] < 1e-12) { w.fill(0); w[i] = 1; return w; }
    }
    for (i = 0; i < n; i++) {
      var a = poly[i], b = poly[(i + 1) % n];
      var ax = a[0] - p[0], ay = a[1] - p[1], bx = b[0] - p[0], by = b[1] - p[1];
      var ang = Math.atan2(ax * by - ay * bx, ax * bx + ay * by);
      t[i] = Math.tan(ang / 2);
    }
    var s = 0;
    for (i = 0; i < n; i++) { w[i] = (t[(i - 1 + n) % n] + t[i]) / r[i]; s += w[i]; }
    for (i = 0; i < n; i++) w[i] /= s;
    return w;
  }

  function inside(p, poly) {
    var c = false;
    for (var i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      var a = poly[i], b = poly[j];
      if ((a[1] > p[1]) !== (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]) c = !c;
    }
    return c;
  }

  function segX(p1, p2, p3, p4) {
    function o(a, b, c) { return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]); }
    var d1 = o(p3, p4, p1), d2 = o(p3, p4, p2), d3 = o(p1, p2, p3), d4 = o(p1, p2, p4);
    return ((d1 > 0) !== (d2 > 0)) && ((d3 > 0) !== (d4 > 0));
  }

  function selfIntersections(poly) {
    var n = poly.length, c = 0;
    for (var i = 0; i < n; i++)
      for (var j = i + 2; j < n; j++) {
        if (i === 0 && j === n - 1) continue;
        if (segX(poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n])) c++;
      }
    return c;
  }

  // ---------- scene ----------
  // A fingertip-like blob: rounded superellipse, tapered toward the tip (bottom).
  function shapeR(th) {
    var c = Math.cos(th), sn = Math.sin(th), a = 0.6, b = 0.5, e = 4;
    var r = Math.pow(Math.pow(Math.abs(c / a), e) + Math.pow(Math.abs(sn / b), e), -1 / e);
    return r * (1 + 0.12 * sn);
  }
  var P = 240, N = 20, M = 7;
  var shape0 = [], cage0 = [], outer = [];
  for (var i = 0; i < P; i++) {
    var th = 2 * Math.PI * i / P, r = shapeR(th);
    shape0.push([r * Math.cos(th), r * Math.sin(th)]);
  }
  var margin = 0.07;
  for (i = 0; i < N; i++) {
    th = 2 * Math.PI * (i + 0.5) / N;
    r = shapeR(th) + margin;
    cage0.push([r * Math.cos(th), r * Math.sin(th)]);
  }
  // grow the cage until it strictly contains the shape
  for (var tries = 0; tries < 20 && shape0.some(function (p) { return !inside(p, cage0); }); tries++)
    cage0 = cage0.map(function (p) { return [p[0] * 1.02, p[1] * 1.02]; });
  var maxR = Math.max.apply(null, cage0.map(function (p) { return Math.hypot(p[0], p[1]); }));
  var outerR = (maxR + 0.12) / Math.cos(Math.PI / M);
  for (i = 0; i < M; i++) {
    th = 2 * Math.PI * i / M + Math.PI / 2;
    outer.push([outerR * Math.cos(th) * 1.08, outerR * Math.sin(th)]);
  }

  // grid lines inside the rest shape for distortion visualization
  var grid = [], step = 0.085, ext = 0.75;
  for (var g = -ext; g <= ext + 1e-9; g += step) {
    var h = [], v = [];
    for (var s = -ext; s <= ext + 1e-9; s += step / 3) { h.push([s, g]); v.push([g, s]); }
    [h, v].forEach(function (line) {
      var pts = line.filter(function (p) { return inside(p, cage0); });
      if (pts.length > 2) grid.push(pts);
    });
  }

  // precompute weights
  var phiShape = shape0.map(function (p) { return mvc(p, cage0); });
  var phiGrid = grid.map(function (l) { return l.map(function (p) { return mvc(p, cage0); }); });
  var W = cage0.map(function (p) { return mvc(p, outer); });
  var Cdense = [];
  for (i = 0; i < N; i++) {
    Cdense.push(new Float64Array(N));
    for (var j = 0; j < N; j++) {
      var c = 0;
      for (var k = 0; k < M; k++) c += Math.min(W[i][k], W[j][k]);
      Cdense[i][j] = c;
    }
  }
  var offVals = [];
  for (i = 0; i < N; i++) for (j = i + 1; j < N; j++) offVals.push(Cdense[i][j]);
  offVals.sort(function (a, b) { return b - a; });

  var state = { alpha: 0.5, dens: 0.4, D: cage0.map(function () { return [0, 0]; }), hover: -1, showOuter: true };
  var C = null;

  function buildC() {
    // keep unit diagonal + largest off-diagonal (symmetric) entries up to density d
    var keep = Math.round(state.dens * offVals.length);
    var thr = keep > 0 ? offVals[keep - 1] : Infinity;
    C = Cdense.map(function (row, i) {
      return Float64Array.from(row, function (v, j) { return i === j ? 1 : (v >= thr ? v : 0); });
    });
  }

  function effectiveD(useSA) {
    var D = state.D;
    if (!useSA) return D;
    return D.map(function (dj, j) {
      var x = dj[0], y = dj[1];
      for (var i = 0; i < N; i++) {
        if (i === j || C[i][j] === 0) continue;
        x += state.alpha * C[i][j] * D[i][0];
        y += state.alpha * C[i][j] * D[i][1];
      }
      return [x, y];
    });
  }

  function deform(useSA) {
    var dD = effectiveD(useSA);
    var cage = cage0.map(function (p, i) { return [p[0] + dD[i][0], p[1] + dD[i][1]]; });
    function apply(phi) {
      var x = 0, y = 0;
      for (var i = 0; i < N; i++) { x += phi[i] * cage[i][0]; y += phi[i] * cage[i][1]; }
      return [x, y];
    }
    return {
      cage: cage,
      shape: phiShape.map(apply),
      grid: phiGrid.map(function (l) { return l.map(apply); })
    };
  }

  function distortion(shape) {
    var e = 0;
    for (var i = 0; i < P; i++) {
      var a = (i - 1 + P) % P, b = (i + 1) % P;
      var lx = shape[i][0] - 0.5 * (shape[a][0] + shape[b][0]) - (shape0[i][0] - 0.5 * (shape0[a][0] + shape0[b][0]));
      var ly = shape[i][1] - 0.5 * (shape[a][1] + shape[b][1]) - (shape0[i][1] - 0.5 * (shape0[a][1] + shape0[b][1]));
      e += lx * lx + ly * ly;
    }
    return 1e7 * e / P;
  }

  // ---------- rendering ----------
  var css = getComputedStyle(document.documentElement);
  function col(name, fb) { var v = css.getPropertyValue(name).trim(); return v || fb; }
  var COL = {
    cbd: col('--cbd', '#e5812b'), sa: col('--sa', '#0f9d7a'),
    ink: col('--ink', '#1d2433'), muted: col('--muted', '#8a93a6')
  };

  function Panel(canvas, useSA) {
    this.cv = canvas; this.ctx = canvas.getContext('2d'); this.useSA = useSA;
    this.view = { s: 1, ox: 0, oy: 0 };
    this.geo = null;
    this.drag = -1;
    var self = this;
    canvas.addEventListener('pointerdown', function (e) { self.down(e); });
    canvas.addEventListener('pointermove', function (e) { self.move(e); });
    canvas.addEventListener('pointerup', function (e) { self.up(e); });
    canvas.addEventListener('pointercancel', function (e) { self.up(e); });
    canvas.addEventListener('pointerleave', function () { if (self.drag < 0 && state.hover !== -1) { state.hover = -1; redraw(); } });
  }
  Panel.prototype.resize = function () {
    var dpr = window.devicePixelRatio || 1, w = this.cv.clientWidth, h = this.cv.clientHeight;
    this.cv.width = Math.round(w * dpr); this.cv.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var span = outerR * 2.35;
    this.view.s = Math.min(w, h) / span;
    this.view.ox = w / 2; this.view.oy = h / 2;
  };
  Panel.prototype.toS = function (p) { return [this.view.ox + p[0] * this.view.s, this.view.oy - p[1] * this.view.s]; };
  Panel.prototype.toW = function (x, y) { return [(x - this.view.ox) / this.view.s, (this.view.oy - y) / this.view.s]; };
  Panel.prototype.pos = function (e) { var b = this.cv.getBoundingClientRect(); return [e.clientX - b.left, e.clientY - b.top]; };
  Panel.prototype.hit = function (xy) {
    if (!this.geo) return -1;
    var best = -1, bd = 16;
    for (var i = 0; i < N; i++) {
      var s = this.toS(this.geo.cage[i]), d = Math.hypot(s[0] - xy[0], s[1] - xy[1]);
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  };
  Panel.prototype.down = function (e) {
    var i = this.hit(this.pos(e));
    if (i < 0) return;
    stopWalk();
    this.drag = i; state.hover = i;
    this.cv.setPointerCapture(e.pointerId);
    this.cv.classList.add('is-dragging');
    e.preventDefault();
  };
  Panel.prototype.move = function (e) {
    var xy = this.pos(e);
    if (this.drag >= 0) {
      var i = this.drag, w = this.toW(xy[0], xy[1]);
      // choose D_i so that *this* panel's handle sits under the cursor
      var x = w[0] - cage0[i][0], y = w[1] - cage0[i][1];
      if (this.useSA) {
        for (var l = 0; l < N; l++) {
          if (l === i || C[l][i] === 0) continue;
          x -= state.alpha * C[l][i] * state.D[l][0];
          y -= state.alpha * C[l][i] * state.D[l][1];
        }
      }
      state.D[i] = [x, y];
      redraw();
    } else {
      var hv = this.hit(xy);
      this.cv.style.cursor = hv >= 0 ? 'grab' : 'default';
      if (hv !== state.hover) { state.hover = hv; redraw(); }
    }
  };
  Panel.prototype.up = function () {
    this.drag = -1; this.cv.classList.remove('is-dragging');
  };
  Panel.prototype.draw = function () {
    var ctx = this.ctx, w = this.cv.clientWidth, h = this.cv.clientHeight, self = this;
    var geo = this.geo = deform(this.useSA);
    var accent = this.useSA ? COL.sa : COL.cbd;
    ctx.clearRect(0, 0, w, h);

    function path(pts, close) {
      ctx.beginPath();
      pts.forEach(function (p, i) { var s = self.toS(p); if (i) ctx.lineTo(s[0], s[1]); else ctx.moveTo(s[0], s[1]); });
      if (close) ctx.closePath();
    }

    if (this.useSA && state.showOuter) {
      ctx.save();
      ctx.setLineDash([5, 5]); ctx.strokeStyle = COL.muted; ctx.lineWidth = 1.2;
      path(outer, true); ctx.stroke(); ctx.setLineDash([]);
      outer.forEach(function (p) { var s = self.toS(p); ctx.fillStyle = COL.muted; ctx.fillRect(s[0] - 3, s[1] - 3, 6, 6); });
      ctx.font = '11px "Noto Sans", sans-serif'; ctx.fillStyle = COL.muted;
      var lp = self.toS(outer[0]); ctx.fillText('outer cage', lp[0] + 8, lp[1] + 4);
      ctx.restore();
    }

    // shape fill + grid clipped to the deformed outline
    ctx.save();
    path(geo.shape, true);
    var grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, '#ffe066'); grad.addColorStop(1, '#f7b733');
    ctx.fillStyle = grad; ctx.fill('evenodd');
    ctx.clip('evenodd');
    ctx.strokeStyle = 'rgba(90,60,0,0.22)'; ctx.lineWidth = 1;
    geo.grid.forEach(function (l) { path(l, false); ctx.stroke(); });
    ctx.restore();
    path(geo.shape, true); ctx.strokeStyle = '#8a5a00'; ctx.lineWidth = 1.6; ctx.stroke();

    // cage
    path(geo.cage, true); ctx.strokeStyle = accent; ctx.globalAlpha = 0.85; ctx.lineWidth = 1.6; ctx.stroke(); ctx.globalAlpha = 1;

    // handles, coloured by affinity to the hovered handle
    var hv = state.hover;
    for (var i = 0; i < N; i++) {
      var s = this.toS(geo.cage[i]);
      var a = hv < 0 ? 0 : (i === hv ? 1 : (this.useSA ? C[hv][i] * state.alpha : 0));
      if (hv >= 0 && i !== hv && a > 0) {
        ctx.beginPath(); ctx.arc(s[0], s[1], 6 + 14 * a, 0, 2 * Math.PI);
        ctx.fillStyle = hexA(accent, 0.12 + 0.35 * a); ctx.fill();
      }
      ctx.beginPath(); ctx.arc(s[0], s[1], i === hv ? 7.5 : 5.5, 0, 2 * Math.PI);
      ctx.fillStyle = i === hv ? accent : '#fff';
      ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = accent; ctx.stroke();
    }
  };

  function hexA(hex, a) {
    var m = /^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i.exec(hex);
    if (!m) return hex;
    return 'rgba(' + parseInt(m[1], 16) + ',' + parseInt(m[2], 16) + ',' + parseInt(m[3], 16) + ',' + a + ')';
  }

  // ---------- affinity matrix view ----------
  function drawMatrix(cv) {
    var dpr = window.devicePixelRatio || 1, w = cv.clientWidth;
    cv.width = Math.round(w * dpr); cv.height = Math.round(w * dpr);
    var ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var cs = w / N;
    for (var i = 0; i < N; i++) for (var j = 0; j < N; j++) {
      var v = C[i][j];
      ctx.fillStyle = v === 0 ? '#f1f3f7' : hexA(COL.sa, 0.08 + 0.92 * v);
      ctx.fillRect(j * cs, i * cs, cs - 0.5, cs - 0.5);
    }
    if (state.hover >= 0) {
      ctx.strokeStyle = COL.ink; ctx.lineWidth = 2;
      ctx.strokeRect(1, state.hover * cs + 1, w - 2, cs - 2);
    }
  }

  // ---------- wiring ----------
  var panels = [], matCv, raf = 0, walkTimer = 0;
  function redraw() {
    if (raf) return;
    raf = requestAnimationFrame(function () {
      raf = 0;
      panels.forEach(function (p) { p.draw(); });
      drawMatrix(matCv);
      var ids = [['m-cbd-x', 'm-cbd-d'], ['m-sa-x', 'm-sa-d']], dist = [];
      panels.forEach(function (p, k) {
        var x = selfIntersections(p.geo.cage), d = distortion(p.geo.shape);
        var ex = document.getElementById(ids[k][0]);
        ex.textContent = x; ex.classList.toggle('is-bad', x > 0);
        dist.push(d);
      });
      dist.forEach(function (d, k) {
        document.getElementById(ids[k][1]).textContent = d < 10 ? d.toFixed(2) : d.toFixed(1);
      });
      var badge = document.getElementById('pg-ratio'), ratio = dist[1] > 0 ? dist[0] / dist[1] : 0;
      badge.hidden = !(dist[0] > 0.05 && ratio >= 1.2);
      badge.textContent = ratio.toFixed(1) + '\u00d7 smoother';
    });
  }

  function randn() { var u = 1 - Math.random(), v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  function kick() {
    stopWalk();
    state.D = cage0.map(function () { return [0, 0]; });
    var picks = [], count = 3 + Math.floor(Math.random() * 2);
    while (picks.length < count) { var i = Math.floor(Math.random() * N); if (picks.indexOf(i) < 0) picks.push(i); }
    picks.forEach(function (i) {
      var p = cage0[i], n = Math.hypot(p[0], p[1]);
      var out = 0.22 + 0.3 * Math.random();
      var sign = Math.random() < 0.35 ? -0.8 : 1;
      state.D[i] = [p[0] / n * out * sign + 0.1 * randn(), p[1] / n * out * sign + 0.1 * randn()];
    });
    redraw();
  }

  function pull() {
    stopWalk();
    state.D = cage0.map(function () { return [0, 0]; });
    var i = Math.floor(N * 0.8), p = cage0[i], n = Math.hypot(p[0], p[1]);
    state.D[i] = [p[0] / n * 0.45 + 0.12, p[1] / n * 0.45];
    state.hover = i;
    redraw();
  }

  function stopWalk() {
    if (!walkTimer) return;
    cancelAnimationFrame(walkTimer); walkTimer = 0;
    var b = document.getElementById('b-anim');
    b.querySelector('i').className = 'fas fa-play'; b.querySelector('span:last-child').textContent = 'Random walk';
  }
  function toggleWalk() {
    var b = document.getElementById('b-anim');
    if (walkTimer) { stopWalk(); return; }
    b.querySelector('i').className = 'fas fa-pause'; b.querySelector('span:last-child').textContent = 'Pause';
    // smooth random walk: each handle follows its own Ornstein-Uhlenbeck target
    var vel = cage0.map(function () { return [0, 0]; });
    (function tick() {
      state.D = state.D.map(function (d, i) {
        vel[i][0] = 0.9 * vel[i][0] + 0.006 * randn() - 0.01 * d[0];
        vel[i][1] = 0.9 * vel[i][1] + 0.006 * randn() - 0.01 * d[1];
        return [d[0] + vel[i][0], d[1] + vel[i][1]];
      });
      redraw();
      walkTimer = requestAnimationFrame(tick);
    })();
  }

  function init() {
    var cbd = document.getElementById('pg-cbd');
    if (!cbd) return;
    panels = [new Panel(cbd, false), new Panel(document.getElementById('pg-sa'), true)];
    matCv = document.getElementById('pg-mat');
    buildC();
    function resize() { panels.forEach(function (p) { p.resize(); }); redraw(); }
    window.addEventListener('resize', resize);
    resize();

    var sa = document.getElementById('s-alpha'), sd = document.getElementById('s-dens');
    sa.addEventListener('input', function () { state.alpha = +sa.value; document.getElementById('o-alpha').textContent = state.alpha.toFixed(2); redraw(); });
    sd.addEventListener('input', function () { state.dens = +sd.value; document.getElementById('o-dens').textContent = state.dens.toFixed(2); buildC(); redraw(); });
    document.getElementById('b-kick').addEventListener('click', kick);
    document.getElementById('b-pull').addEventListener('click', pull);
    document.getElementById('b-anim').addEventListener('click', toggleWalk);
    document.getElementById('b-reset').addEventListener('click', function () { stopWalk(); state.D = cage0.map(function () { return [0, 0]; }); redraw(); });
    document.getElementById('c-outer').addEventListener('change', function (e) { state.showOuter = e.target.checked; redraw(); });

    matCv.addEventListener('pointermove', function (e) {
      var b = matCv.getBoundingClientRect(), i = Math.floor((e.clientY - b.top) / (b.height / N));
      if (i >= 0 && i < N && i !== state.hover) { state.hover = i; redraw(); }
    });
    matCv.addEventListener('pointerleave', function () { state.hover = -1; redraw(); });

    // open on a deformation that makes the point
    pull();
  }

  // exported for tests
  window.SAPlayground = { mvc: mvc, Cdense: Cdense, W: W, state: state, deform: function (sa) { return deform(sa); }, distortion: distortion, selfIntersections: selfIntersections, buildC: buildC };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
