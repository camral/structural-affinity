(function () {
  'use strict';
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var reduceMotion = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---------- Table 1 (Neural Cages), six seeds, mean & sample SD ----------
  var NC = {
    methods: ['Vanilla NC', 'CBD-SA NoReg', 'CBD-SA NC Loss'],
    cats: ['Chair', 'Table', 'Car'],
    metrics: [
      { key: 'cagelap', label: 'Cage Laplacian', unit: '×10³', data: {
        Chair: [[9.69, 1.70], [1.70, 0.35], [1.55, 0.17]],
        Table: [[14.65, 1.53], [3.36, 0.54], [2.75, 0.61]],
        Car: [[0.97, 0.30], [0.21, 0.02], [0.15, 0.02]] } },
      { key: 'cageedge', label: 'Cage edge', unit: '', data: {
        Chair: [[0.285, 0.018], [0.110, 0.007], [0.106, 0.006]],
        Table: [[0.346, 0.024], [0.180, 0.013], [0.164, 0.006]],
        Car: [[0.074, 0.009], [0.033, 0.002], [0.026, 0.002]] } },
      { key: 'selfx', label: 'Self-intersections', unit: '×10⁴', data: {
        Chair: [[0.53, 0.52], [0, 0], [0, 0]],
        Table: [[11.19, 3.46], [0.29, 0.50], [0, 0]],
        Car: [[0, 0], [0, 0], [0, 0]] } },
      { key: 'cotlap', label: 'Cot-Laplacian', unit: '', note: 'Car values ×10⁻⁴.', data: {
        Chair: [[10.32, 3.53], [9.38, 4.13], [7.62, 2.85]],
        Table: [[145.75, 188.79], [70.46, 84.08], [44.64, 46.89]],
        Car: [[2.68, 1.13], [1.58, 0.38], [0.97, 0.30]] } },
      { key: 'dcd', label: 'DCD (fidelity)', unit: '', data: {
        Chair: [[0.9100, 0.0056], [0.9153, 0.0050], [0.9144, 0.0051]],
        Table: [[0.8716, 0.0101], [0.8790, 0.0090], [0.8774, 0.0116]],
        Car: [[0.8341, 0.0057], [0.8309, 0.0056], [0.8340, 0.0058]] } }
    ]
  };

  // ---------- refined_convex38_paired.csv (matched valid pairs only) ----------
  var AERO = {
    5: [[10, .029824441, .027906450, .027642771], [11, .029481965, .027953571, .026870066], [12, .029909588, .027906729, .027699512],
        [13, .030001908, .026967703, .026782299], [14, .030471980, .026908864, .026708322], [16, .030019409, .027892393, .026911400],
        [17, .029161808, .027899273, .027716109], [19, .029075133, .027822359, .027551670]],
    10: [[11, .030279951, .027306997, .027096363], [12, .030633184, .027950847, .027780339], [13, .030695444, .026959080, .026771699],
         [14, .032366597, .026901077, .026696399], [15, .030672542, .027011555, .026817957], [16, .030746044, .027898575, .027702773],
         [17, .028948371, .027917917, .027727984], [18, .030037825, .027856944, .027635771], [19, .029370974, .027800002, .027505505]]
  };

  function svgEl(tag, attrs, parent) {
    var e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function fmt(v) {
    if (v === 0) return '0';
    var a = Math.abs(v);
    return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : v.toFixed(3);
  }

  // ---------- Neural Cages small multiples ----------
  function ncChart() {
    var btns = $('#nc-metric-btns'), mult = $('#nc-multiples');
    if (!btns) return;
    var current = 0;
    NC.metrics.forEach(function (m, i) {
      var b = document.createElement('button');
      b.className = 'nc-btn' + (i === 0 ? ' is-active' : '');
      b.textContent = m.label;
      b.addEventListener('click', function () {
        current = i;
        $$('.nc-btn', btns).forEach(function (x, j) { x.classList.toggle('is-active', j === i); });
        render();
      });
      btns.appendChild(b);
    });
    var cls = ['van', 'noreg', 'loss'];
    function render() {
      var m = NC.metrics[current];
      mult.innerHTML = '';
      NC.cats.forEach(function (cat) {
        var d = m.data[cat];
        var box = document.createElement('div'); box.className = 'nc-mult';
        var W = 220, H = 190, top = 26, bot = 26, left = 10;
        var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': m.label + ' for ' + cat }, box);
        var max = Math.max.apply(null, d.map(function (x) { return x[0] + x[1]; })) || 1;
        // DCD is a fidelity metric with a narrow range; zoom the axis so differences are visible but honest-labelled
        var base = m.key === 'dcd' ? Math.min.apply(null, d.map(function (x) { return x[0] - x[1]; })) * 0.98 : 0;
        var y = function (v) { return top + (H - top - bot) * (1 - (v - base) / (max - base)); };
        var bw = 46, gap = 22, x0 = left + (W - left - (3 * bw + 2 * gap)) / 2;
        svgEl('line', { x1: 4, x2: W - 4, y1: H - bot, y2: H - bot, class: 'nc-axis' }, svg);
        d.forEach(function (v, k) {
          var x = x0 + k * (bw + gap), yv = y(v[0]);
          var r = svgEl('rect', { x: x, width: bw, y: H - bot, height: 0, rx: 4, class: 'nc-bar nc-' + cls[k] }, svg);
          var t = svgEl('title', {}, r); t.textContent = NC.methods[k] + ': ' + fmt(v[0]) + ' ± ' + fmt(v[1]);
          requestAnimationFrame(function () {
            r.setAttribute('y', Math.min(yv, H - bot - (v[0] > 0 ? 1.5 : 0)));
            r.setAttribute('height', Math.max(H - bot - yv, v[0] > 0 ? 1.5 : 0));
          });
          if (v[1] > 0) {
            var cx = x + bw / 2;
            svgEl('line', { x1: cx, x2: cx, y1: y(v[0] + v[1]), y2: y(Math.max(base, v[0] - v[1])), class: 'nc-err' }, svg);
          }
          var lab = svgEl('text', { x: x + bw / 2, y: Math.max(12, y(v[0] + v[1]) - 5), class: 'nc-val' }, svg);
          if (k === 0) lab.textContent = fmt(v[0]);
          else if (d[0][0] === 0) lab.textContent = fmt(v[0]);
          else {
            var pct = 100 * (v[0] - d[0][0]) / d[0][0];
            lab.textContent = (pct > 0 ? '+' : '−') + Math.abs(pct).toFixed(pct > -10 && pct < 10 ? 1 : 0) + '%';
            lab.setAttribute('class', 'nc-val ' + (pct < 0 ? 'is-good' : 'is-worse'));
          }
        });
        var cap = document.createElement('div'); cap.className = 'nc-cat'; cap.textContent = cat;
        box.appendChild(cap);
        mult.appendChild(box);
      });
      var note = 'Mean ± sample SD over six seeds. Lower is better. Labels show change vs. Vanilla NC.';
      if (m.unit) note += ' Values ' + m.unit + '.';
      if (m.note) note += ' ' + m.note;
      if (m.key === 'dcd') note += ' Axis zoomed: all three methods are within 1%.';
      if (m.key === 'selfx') note += ' Car cages never self-intersect.';
      $('#nc-note').textContent = note;
    }
    render();
  }

  // ---------- Aero paired slope chart ----------
  function aeroChart() {
    var svg = $('#aero-svg'), out = $('#aero-readout');
    if (!svg) return;
    var pert = 5;
    function render() {
      svg.innerHTML = '';
      var rows = AERO[pert];
      var W = 640, H = 340, L = 70, R = 70, T = 20, B = 40;
      var xs = [L + 40, (L + W - R) / 2, W - R - 40], names = ['Initial', 'CBD', 'CBD-SA'];
      var all = []; rows.forEach(function (r) { all.push(r[1], r[2], r[3]); });
      var lo = Math.min.apply(null, all) - 0.0002, hi = Math.max.apply(null, all) + 0.0002;
      var y = function (v) { return T + (H - T - B) * (1 - (v - lo) / (hi - lo)); };
      for (var t = Math.ceil(lo * 1000) / 1000; t <= hi; t += 0.001) {
        svgEl('line', { x1: L, x2: W - R, y1: y(t), y2: y(t), class: 'ae-grid' }, svg);
        svgEl('text', { x: L - 8, y: y(t) + 4, class: 'ae-tick' }, svg).textContent = t.toFixed(3);
      }
      svgEl('text', { x: 16, y: (H - B + T) / 2, class: 'ae-ylab', transform: 'rotate(-90 16 ' + (H - B + T) / 2 + ')' }, svg).textContent = 'drag coefficient C_D';
      names.forEach(function (n, i) {
        svgEl('text', { x: xs[i], y: H - 12, class: 'ae-xlab ' + (i === 1 ? 'c-cbd' : i === 2 ? 'c-sa' : '') }, svg).textContent = n;
      });
      var groups = rows.map(function (r) {
        var g = svgEl('g', { class: 'ae-run', tabindex: 0 }, svg);
        svgEl('polyline', { points: xs[0] + ',' + y(r[1]) + ' ' + xs[1] + ',' + y(r[2]), class: 'ae-l0' }, g);
        svgEl('polyline', { points: xs[1] + ',' + y(r[2]) + ' ' + xs[2] + ',' + y(r[3]), class: 'ae-l1' }, g);
        svgEl('circle', { cx: xs[0], cy: y(r[1]), r: 3.5, class: 'ae-p0' }, g);
        svgEl('circle', { cx: xs[1], cy: y(r[2]), r: 4, class: 'ae-p1' }, g);
        svgEl('circle', { cx: xs[2], cy: y(r[3]), r: 4, class: 'ae-p2' }, g);
        function show() {
          groups.forEach(function (o) { o.classList.toggle('is-dim', o !== g); });
          var red = 100 * (r[2] - r[3]) / r[2];
          out.innerHTML = 'Seed ' + r[0] + ': initial <b>' + r[1].toFixed(5) + '</b> &rarr; CBD <b class="c-cbd">' + r[2].toFixed(5) +
            '</b> &rarr; CBD-SA <b class="c-sa">' + r[3].toFixed(5) + '</b> &nbsp;(' + red.toFixed(2) + '% lower with SA)';
        }
        g.addEventListener('mouseenter', show); g.addEventListener('focus', show);
        g.addEventListener('mouseleave', summary); g.addEventListener('blur', summary);
        return g;
      });
      var mean = function (k) { return rows.reduce(function (s, r) { return s + r[k]; }, 0) / rows.length; };
      var mi = mean(1), mc = mean(2), ms = mean(3);
      var mg = svgEl('g', { class: 'ae-mean' }, svg);
      svgEl('polyline', { points: xs[1] + ',' + y(mc) + ' ' + xs[2] + ',' + y(ms) }, mg);
      svgEl('circle', { cx: xs[1], cy: y(mc), r: 5.5 }, mg);
      svgEl('circle', { cx: xs[2], cy: y(ms), r: 5.5 }, mg);
      svgEl('text', { x: xs[1] - 12, y: y(mc) + 4, 'text-anchor': 'end', class: 'ae-mean-lab' }, mg).textContent = 'mean ' + mc.toFixed(5);
      svgEl('text', { x: xs[2] + 10, y: y(ms) + 4, 'text-anchor': 'start', class: 'ae-mean-lab' }, mg).textContent = ms.toFixed(5);
      // callout in the empty upper-right region, with a leader to the mean line's midpoint
      var mx = (xs[1] + xs[2]) / 2, my = (y(mc) + y(ms)) / 2;
      var bx = xs[2] + 20, by = T + 34;
      var call = svgEl('g', { class: 'ae-callout' }, mg);
      svgEl('line', { x1: mx, y1: by + 30, x2: mx, y2: my - 8, class: 'ae-leader' }, call);
      var t1 = svgEl('text', { x: bx, y: by, 'text-anchor': 'end', class: 'ae-call-h' }, call);
      t1.textContent = 'Thick line = mean of ' + rows.length + ' seeds';
      var t2 = svgEl('text', { x: bx, y: by + 20, 'text-anchor': 'end', class: 'ae-call' }, call);
      t2.textContent = 'CBD ' + mc.toFixed(5) + ' \u2192 CBD-SA ' + ms.toFixed(5) + '  (\u2212' + (100 * (mc - ms) / mc).toFixed(2) + '%)';
      function summary() {
        groups.forEach(function (o) { o.classList.remove('is-dim'); });
        var wins = rows.filter(function (r) { return r[3] < r[2]; }).length;
        var gain = 100 * (mc - ms) / (mi - mc);
        out.innerHTML = 'CBD-SA lower in <b class="c-sa">' + wins + '/' + rows.length + '</b> pairs &middot; mean C<sub>D</sub> &minus;<b>' +
          (100 * (mc - ms) / mc).toFixed(2) + '%</b> &middot; <b>+' + gain.toFixed(1) + '%</b> extra optimization gain over CBD';
      }
      summary();
    }
    $$('.aero-pert li').forEach(function (li) {
      li.addEventListener('click', function () {
        $$('.aero-pert li').forEach(function (x) { x.classList.toggle('is-active', x === li); });
        pert = +li.dataset.pert; render();
      });
    });
    render();
  }

  // ---------- spatula videos: synced playback + frame scrubber ----------
  function spatula() {
    var range = $('#sp-range');
    if (!range) return;
    var a = $('#sp-a'), b = $('#sp-b'), t = $('#sp-t'), play = $('#sp-play'), icon = play.querySelector('i');
    function label() { t.textContent = 't = ' + a.currentTime.toFixed(2) + ' s'; }
    function setIcon() {
      var paused = a.paused;
      icon.className = paused ? 'fas fa-play' : 'fas fa-pause';
      play.setAttribute('aria-label', paused ? 'Play' : 'Pause');
    }
    function setMax() { if (a.duration) range.max = a.duration.toFixed(3); }
    setMax(); a.addEventListener('loadedmetadata', setMax);       // metadata may already be loaded (cache)
    (function tick() {                                   // keep slider, label and the CBD video on the CBD-SA clock
      if (!a.paused) {
        range.value = a.currentTime;
        if (!b.seeking && Math.abs(b.currentTime - a.currentTime) > 0.15) b.currentTime = a.currentTime;
        label();
      }
      requestAnimationFrame(tick);
    })();
    range.addEventListener('input', function () {
      a.pause(); b.pause(); setIcon();
      a.currentTime = b.currentTime = +range.value;
      label();
    });
    play.addEventListener('click', function () {
      if (a.paused) { b.currentTime = a.currentTime; a.play(); b.play(); } else { a.pause(); b.pause(); }
      setIcon();
    });
    a.addEventListener('play', setIcon); a.addEventListener('pause', setIcon);
  }

  // ---------- CBD / CBD-SA switch on combined figure ----------
  function switches() {
    $$('.sa-switch').forEach(function (sw) {
      var crop = $('.sa-crop', sw), cap = $('.sa-switch-cap', sw);
      var pos = {}; crop.dataset.sidePos.split(',').forEach(function (kv) { var p = kv.split(':'); pos[p[0]] = p[1]; });
      function set(side) {
        $$('.sa-switch-btn', sw).forEach(function (b) { b.classList.toggle('is-active', b.dataset.side === side); });
        crop.style.backgroundPosition = pos[side] + ' 0';
        crop.classList.toggle('is-sa', side === 'sa');
        cap.textContent = cap.dataset['cap' + side.charAt(0).toUpperCase() + side.slice(1)];
      }
      $$('.sa-switch-btn', sw).forEach(function (b) { b.addEventListener('click', function () { set(b.dataset.side); }); });
      set('cbd');
    });
  }

  // ---------- counters, bars, reveal ----------
  function countUp(el) {
    var target = +el.dataset.count, dec = +(el.dataset.decimals || 0), t0 = null, dur = 1100;
    if (reduceMotion) { el.textContent = target.toFixed(dec); return; }
    (function step(ts) {
      if (!t0) t0 = ts;
      var k = Math.min(1, (ts - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      el.textContent = (target * e).toFixed(dec);
      if (k < 1) requestAnimationFrame(step);
    })(performance.now());
  }
  function reveal() {
    var targets = $$('[data-reveal], [data-count], .sa-bigbars');
    function fire(el) {
      el.classList.add('is-in');
      if (el.dataset.count) countUp(el);
      $$('[data-count]', el).forEach(function (c) { if (!c.dataset.done) { c.dataset.done = 1; countUp(c); } });
      $$('.sa-bb-bar', el).forEach(function (b) { b.style.width = b.dataset.w + '%'; });
    }
    if (!('IntersectionObserver' in window)) { targets.forEach(fire); return; }
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { fire(e.target); io.unobserve(e.target); } });
    }, { threshold: 0.08, rootMargin: "0px 0px -5% 0px" });
    targets.forEach(function (t) { if (t.dataset.count) return; io.observe(t); });
  }

  // ---------- tabs ----------
  function tabs() {
    $$('.sa-tabs li').forEach(function (li) {
      li.addEventListener('click', function () {
        $$('.sa-tabs li').forEach(function (x) { x.classList.toggle('is-active', x === li); });
        $$('.sa-tab-pane').forEach(function (p) { p.classList.toggle('is-active', p.id === 'tab-' + li.dataset.tab); });
        $$('#tab-' + li.dataset.tab + ' .sa-bigbars, #tab-' + li.dataset.tab + ' [data-reveal]').forEach(function (el) {
          el.classList.add('is-in');
          $$('.sa-bb-bar', el).forEach(function (b) { b.style.width = b.dataset.w + '%'; });
        });
      });
    });
  }

  // ---------- misc ----------
  function lightbox() {
    var lb = $('#lightbox'), img = $('img', lb);
    $$('.sa-zoom img').forEach(function (im) {
      im.addEventListener('click', function () { img.src = im.src; img.alt = im.alt; lb.hidden = false; });
    });
    lb.addEventListener('click', function () { lb.hidden = true; });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') lb.hidden = true; });
  }
  function bib() {
    var b = $('#bib-copy');
    if (!b) return;
    b.addEventListener('click', function () {
      var txt = $('#bib-text').textContent;
      var done = function () { b.querySelector('span:last-child').textContent = 'Copied!'; setTimeout(function () { b.querySelector('span:last-child').textContent = 'Copy'; }, 1500); };
      if (navigator.clipboard) navigator.clipboard.writeText(txt).then(done, function () {});
    });
  }
  function navbar() {
    $$('.navbar-burger').forEach(function (el) {
      el.addEventListener('click', function () {
        el.classList.toggle('is-active');
        document.getElementById(el.dataset.target).classList.toggle('is-active');
      });
    });
    $$('.navbar-menu a').forEach(function (a) { a.addEventListener('click', function () { $$('.navbar-burger, .navbar-menu').forEach(function (x) { x.classList.remove('is-active'); }); }); });
  }
  function math() {
    if (window.renderMathInElement) {
      renderMathInElement(document.body, { delimiters: [{ left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }], throwOnError: false });
    }
  }

  function init() {
    navbar(); tabs(); ncChart(); aeroChart(); spatula(); switches(); reveal(); lightbox(); bib(); math();
  }
  window.SAData = { NC: NC, AERO: AERO };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
