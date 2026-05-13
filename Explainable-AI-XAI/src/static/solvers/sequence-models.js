// ============================================================================
// Sequence models: vanilla RNN and LSTM.
//
// Canvas points are treated as a 1-D time series: X = time, Y = signal.
// At each step t the model consumes y_t and predicts y_{t+1}.
// Weights are seeded random — these visualisations are about the forward
// recurrence (the mechanism), not about training to minimise loss.
// ============================================================================

function seqRng(seed) {
  let state = (seed | 0) || 1;
  return function () {
    state = (state + 0x6D2B79F5) | 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function seqGauss(rng) {
  // Cheap Irwin-Hall approximation; scaled to ~N(0, 0.25)
  return (rng() + rng() + rng() + rng() - 2) * 0.4;
}

function seqFmt(x, digits = 3) {
  if (x === undefined || x === null || Number.isNaN(x)) return '—';
  if (Math.abs(x) < 1e-9) return '0';
  return Number(x).toFixed(digits);
}

function seqMatrixTable(M, opts = {}) {
  const { rowLabels = null, colLabels = null, digits = 3, caption = null, highlightRow = null } = opts;
  const rows = M.length, cols = rows > 0 ? M[0].length : 0;
  let html = '<div class="table-container"><table>';
  if (caption) html += `<caption style="caption-side:top;font-weight:600;text-align:left;margin-bottom:4px;">${caption}</caption>`;
  if (colLabels) {
    html += '<thead><tr>';
    if (rowLabels) html += '<th></th>';
    for (let j = 0; j < cols; j++) html += `<th>${colLabels[j] !== undefined ? colLabels[j] : j}</th>`;
    html += '</tr></thead>';
  }
  html += '<tbody>';
  for (let i = 0; i < rows; i++) {
    const rowStyle = (highlightRow !== null && i === highlightRow) ? ' style="background:var(--accent-soft);"' : '';
    html += `<tr${rowStyle}>`;
    if (rowLabels) html += `<th>${rowLabels[i] !== undefined ? rowLabels[i] : i}</th>`;
    for (let j = 0; j < cols; j++) html += `<td>${seqFmt(M[i][j], digits)}</td>`;
    html += '</tr>';
  }
  html += '</tbody></table></div>';
  return html;
}

function seqVectorTable(v, labels = null, opts = {}) {
  return seqMatrixTable([v], Object.assign({ colLabels: labels, rowLabels: opts.rowLabel ? [opts.rowLabel] : null }, opts));
}

// Sort points by X and return parallel arrays of x's and y's. Time series only
// makes sense in a defined order, so we always sort.
function seqPrepare(points) {
  const sorted = [...points].sort((a, b) => a.x - b.x);
  return {
    xs: sorted.map(p => p.x),
    ys: sorted.map(p => p.y),
  };
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

// ===========================================================================
// VANILLA RNN
// ===========================================================================
//
//   h_t = tanh(W_x * y_{t-1} + W_h * h_{t-1} + b_h)      (hidden update)
//   ŷ_t = W_y · h_t + b_y                                (output)
//
// Input dimension is 1 (scalar signal). Output dimension is 1.
// W_x: H, W_h: H x H, b_h: H, W_y: H, b_y: scalar.
// ===========================================================================

class VanillaRNN {
  constructor() { this.reset(); }

  reset() {
    this.model = null;
    this.t = 0;          // pointer to the next time step to compute
    this.history = [];   // per-step records: { y_in, h, y_pred, y_target }
  }

  initialize(points, hiddenDim, seed) {
    const seq = seqPrepare(points);
    if (seq.ys.length < 3) {
      window.showToast('Need at least 3 points (sort-by-X gives the time series).', 'warn');
      return false;
    }
    const H = Math.max(1, Math.min(16, hiddenDim | 0));
    const rng = seqRng(seed);

    // Initialise weights (Xavier-ish via seqGauss).
    const Wx = Array.from({ length: H }, () => seqGauss(rng));
    const Wh = Array.from({ length: H }, () => Array.from({ length: H }, () => seqGauss(rng)));
    const bh = new Array(H).fill(0);
    const Wy = Array.from({ length: H }, () => seqGauss(rng));
    const by = 0;

    this.model = {
      fitted: true,
      H,
      seed,
      xs: seq.xs,
      ys: seq.ys,
      Wx, Wh, bh, Wy, by,
      h0: new Array(H).fill(0),
    };
    this.t = 0;
    this.history = [];
    return true;
  }

  // Compute one step using the previous y (y_{t-1}) to produce h_t and ŷ_t.
  // The very first step has no y_{-1}, so we feed y_0 as input and produce h_1,
  // then ŷ_1 is the prediction of y_1. We index history by the *predicted* index.
  nextStep() {
    if (!this.model) return null;
    const m = this.model;
    if (this.t >= m.ys.length - 1) return null;

    const yIn = m.ys[this.t]; // input is the current observation
    const hPrev = this.t === 0 ? m.h0.slice() : this.history[this.t - 1].h.slice();

    // Pre-activation: z = W_x * yIn + W_h · hPrev + b_h
    const z = new Array(m.H);
    for (let i = 0; i < m.H; i++) {
      let s = m.Wx[i] * yIn + m.bh[i];
      for (let j = 0; j < m.H; j++) s += m.Wh[i][j] * hPrev[j];
      z[i] = s;
    }
    const h = z.map(Math.tanh);

    // Output: ŷ = W_y · h + b_y
    let yPred = m.by;
    for (let i = 0; i < m.H; i++) yPred += m.Wy[i] * h[i];

    const yTarget = m.ys[this.t + 1];
    this.history.push({ step: this.t, yIn, hPrev, z, h, yPred, yTarget });
    this.t++;
    return this.history[this.history.length - 1];
  }

  runAll() {
    if (!this.model) return;
    while (this.t < this.model.ys.length - 1) this.nextStep();
  }

  // ---------- DOM rendering ----------

  renderWeights() {
    const m = this.model;
    if (!m) return;
    const out = document.getElementById('rnn-weights');
    if (!out) return;
    const dimLabels = Array.from({ length: m.H }, (_, i) => `h${i}`);
    let html = `<p style="margin:6px 0;font-size:12px;color:var(--fg-muted);">Hidden size H = ${m.H}. Inputs/outputs are scalar.</p>`;
    html += seqMatrixTable([m.Wx], { colLabels: dimLabels, rowLabels: ['W_x'], digits: 3, caption: 'W_x  (1 → H)' });
    html += seqMatrixTable(m.Wh, { rowLabels: dimLabels, colLabels: dimLabels, digits: 3, caption: 'W_h  (H → H)' });
    html += seqMatrixTable([m.bh], { colLabels: dimLabels, rowLabels: ['b_h'], digits: 3 });
    html += seqMatrixTable([m.Wy], { colLabels: dimLabels, rowLabels: ['W_y'], digits: 3, caption: 'W_y  (H → 1),   b_y = ' + seqFmt(m.by, 3) });
    out.innerHTML = html;
    out.parentElement.style.display = 'block';
  }

  renderHistory() {
    const m = this.model;
    const out = document.getElementById('rnn-history');
    if (!out || !m) return;
    if (this.history.length === 0) {
      out.innerHTML = '<p style="font-size:13px;color:var(--fg-muted);">Run a step to populate the table.</p>';
      out.parentElement.style.display = 'block';
      return;
    }
    const hCols = Array.from({ length: m.H }, (_, i) => `h${i}`);
    const headers = ['t', 'y_in (y_t)', ...hCols, 'ŷ_{t+1}', 'target y_{t+1}', 'err'];
    let html = '<div class="table-container"><table><thead><tr>';
    for (const h of headers) html += `<th>${h}</th>`;
    html += '</tr></thead><tbody>';
    for (let i = 0; i < this.history.length; i++) {
      const r = this.history[i];
      const isLast = i === this.history.length - 1;
      html += `<tr${isLast ? ' style="background:var(--accent-soft);"' : ''}><th>${r.step}</th><td>${seqFmt(r.yIn, 3)}</td>`;
      for (let j = 0; j < m.H; j++) html += `<td>${seqFmt(r.h[j], 3)}</td>`;
      html += `<td>${seqFmt(r.yPred, 3)}</td><td>${seqFmt(r.yTarget, 3)}</td><td>${seqFmt(r.yPred - r.yTarget, 3)}</td></tr>`;
    }
    html += '</tbody></table></div>';
    out.innerHTML = html;
    out.parentElement.style.display = 'block';
  }

  renderStep(record) {
    const m = this.model;
    const out = document.getElementById('rnn-step-detail');
    if (!out || !m || !record) return;
    const dimLabels = Array.from({ length: m.H }, (_, i) => `h${i}`);
    let html = `<p style="margin:6px 0;"><strong>Step t = ${record.step}</strong>: feed y_t = <code>${seqFmt(record.yIn, 3)}</code>, predict y_{t+1}.</p>`;
    html += seqMatrixTable([record.hPrev], { colLabels: dimLabels, rowLabels: ['h_{t-1}'], digits: 3 });
    html += seqMatrixTable([record.z], { colLabels: dimLabels, rowLabels: ['z'], digits: 3, caption: 'z = W_x·y_t + W_h·h_{t-1} + b_h' });
    html += seqMatrixTable([record.h], { colLabels: dimLabels, rowLabels: ['h_t'], digits: 3, caption: 'h_t = tanh(z)' });
    html += `<p style="margin:6px 0;">ŷ_{t+1} = W_y · h_t + b_y = <strong>${seqFmt(record.yPred, 4)}</strong>  (target: ${seqFmt(record.yTarget, 4)}, error ${seqFmt(record.yPred - record.yTarget, 4)})</p>`;
    out.innerHTML = html;
    out.parentElement.style.display = 'block';
  }

  renderMetrics() {
    const m = this.model;
    if (!m || this.history.length === 0) return;
    let sse = 0;
    for (const r of this.history) sse += (r.yPred - r.yTarget) * (r.yPred - r.yTarget);
    const mse = sse / this.history.length;
    const el = document.getElementById('rnn-mse');
    if (el) el.textContent = seqFmt(mse, 4);
  }

  // Predictions for the canvas drawer: array of {x, y} marker points.
  predictionMarkers() {
    if (!this.model || this.history.length === 0) return [];
    const m = this.model;
    return this.history.map(r => ({ x: m.xs[r.step + 1], y: r.yPred }));
  }

  // ---------- Training (numerical gradients) ----------

  // Run a full forward pass with the current weights and return mean-squared
  // error over all (T-1) prediction steps. Does NOT touch this.history.
  _computeMSE() {
    const m = this.model;
    if (!m) return Infinity;
    const N = m.ys.length - 1;
    let sse = 0;
    let h = m.h0.slice();
    for (let t = 0; t < N; t++) {
      const yIn = m.ys[t];
      const z = new Array(m.H);
      for (let i = 0; i < m.H; i++) {
        let s = m.Wx[i] * yIn + m.bh[i];
        for (let j = 0; j < m.H; j++) s += m.Wh[i][j] * h[j];
        z[i] = s;
      }
      h = z.map(Math.tanh);
      let yPred = m.by;
      for (let i = 0; i < m.H; i++) yPred += m.Wy[i] * h[i];
      const err = yPred - m.ys[t + 1];
      sse += err * err;
    }
    return sse / N;
  }

  // For each scalar parameter, call cb(getter, setter). cb returns nothing —
  // the train loop uses (Lp - Lm)/(2*eps) to estimate ∂L/∂param and updates
  // in-place.
  _eachParam(cb) {
    const m = this.model;
    // Wx: H scalars
    for (let i = 0; i < m.H; i++) {
      cb(() => m.Wx[i], v => { m.Wx[i] = v; });
    }
    // Wh: H x H
    for (let i = 0; i < m.H; i++) {
      for (let j = 0; j < m.H; j++) {
        cb(() => m.Wh[i][j], v => { m.Wh[i][j] = v; });
      }
    }
    // bh: H
    for (let i = 0; i < m.H; i++) {
      cb(() => m.bh[i], v => { m.bh[i] = v; });
    }
    // Wy: H
    for (let i = 0; i < m.H; i++) {
      cb(() => m.Wy[i], v => { m.Wy[i] = v; });
    }
    // by: scalar
    cb(() => m.by, v => { m.by = v; });
  }

  // Train via numerical gradient descent. Returns the loss history.
  // The math is identical to BPTT — we just compute ∂L/∂θ with central
  // differences instead of unrolling the chain rule by hand.
  train(epochs, lr, eps = 1e-4) {
    if (!this.model) return [];
    const lossHistory = [];
    for (let e = 0; e < epochs; e++) {
      const L0 = this._computeMSE();
      lossHistory.push(L0);
      // Collect gradients first so we update all params with the same baseline.
      const grads = [];
      this._eachParam((get, set) => {
        const v = get();
        set(v + eps);
        const Lp = this._computeMSE();
        set(v - eps);
        const Lm = this._computeMSE();
        set(v);
        grads.push({ set, v, grad: (Lp - Lm) / (2 * eps) });
      });
      // Gradient descent update
      for (const g of grads) g.set(g.v - lr * g.grad);
    }
    // Re-run forward to populate this.history with the trained predictions.
    this.t = 0;
    this.history = [];
    while (this.t < this.model.ys.length - 1) this.nextStep();
    return lossHistory;
  }
}

// ===========================================================================
// LSTM
// ===========================================================================
//
// Concat z_t = [h_{t-1}, y_t] ∈ ℝ^{H+1}.
//   f_t = σ(W_f · z_t + b_f)        forget gate
//   i_t = σ(W_i · z_t + b_i)        input gate
//   g_t = tanh(W_g · z_t + b_g)     candidate cell
//   o_t = σ(W_o · z_t + b_o)        output gate
//   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t   cell state
//   h_t = o_t ⊙ tanh(c_t)              hidden state
//   ŷ_t = W_y · h_t + b_y             prediction
//
// All gate matrices are H × (H+1).
// ===========================================================================

class LSTM {
  constructor() { this.reset(); }

  reset() {
    this.model = null;
    this.t = 0;
    this.history = [];
  }

  initialize(points, hiddenDim, seed) {
    const seq = seqPrepare(points);
    if (seq.ys.length < 3) {
      window.showToast('Need at least 3 points (sort-by-X gives the time series).', 'warn');
      return false;
    }
    const H = Math.max(1, Math.min(16, hiddenDim | 0));
    const rng = seqRng(seed);
    const D = H + 1; // concat dim

    const gateMat = () => Array.from({ length: H }, () => Array.from({ length: D }, () => seqGauss(rng)));
    const Wf = gateMat(), Wi = gateMat(), Wg = gateMat(), Wo = gateMat();
    // Forget gate bias initialised to +1 — a classic LSTM trick that helps
    // the cell retain information at the start.
    const bf = new Array(H).fill(1.0);
    const bi = new Array(H).fill(0);
    const bg = new Array(H).fill(0);
    const bo = new Array(H).fill(0);
    const Wy = Array.from({ length: H }, () => seqGauss(rng));
    const by = 0;

    this.model = {
      fitted: true,
      H,
      seed,
      xs: seq.xs,
      ys: seq.ys,
      Wf, Wi, Wg, Wo, bf, bi, bg, bo, Wy, by,
      h0: new Array(H).fill(0),
      c0: new Array(H).fill(0),
    };
    this.t = 0;
    this.history = [];
    return true;
  }

  _gate(W, b, z) {
    const H = b.length;
    const out = new Array(H);
    for (let i = 0; i < H; i++) {
      let s = b[i];
      for (let j = 0; j < W[i].length; j++) s += W[i][j] * z[j];
      out[i] = s;
    }
    return out;
  }

  nextStep() {
    if (!this.model) return null;
    const m = this.model;
    if (this.t >= m.ys.length - 1) return null;

    const yIn = m.ys[this.t];
    const last = this.t === 0
      ? { h: m.h0.slice(), c: m.c0.slice() }
      : { h: this.history[this.t - 1].h.slice(), c: this.history[this.t - 1].c.slice() };
    const z = last.h.concat([yIn]); // [h_{t-1}, y_t]

    const fPre = this._gate(m.Wf, m.bf, z);
    const iPre = this._gate(m.Wi, m.bi, z);
    const gPre = this._gate(m.Wg, m.bg, z);
    const oPre = this._gate(m.Wo, m.bo, z);
    const f = fPre.map(sigmoid);
    const i = iPre.map(sigmoid);
    const g = gPre.map(Math.tanh);
    const o = oPre.map(sigmoid);

    const cPrev = last.c;
    const c = new Array(m.H);
    for (let k = 0; k < m.H; k++) c[k] = f[k] * cPrev[k] + i[k] * g[k];
    const h = c.map((v, k) => o[k] * Math.tanh(v));

    let yPred = m.by;
    for (let k = 0; k < m.H; k++) yPred += m.Wy[k] * h[k];

    const yTarget = m.ys[this.t + 1];
    this.history.push({ step: this.t, yIn, hPrev: last.h, cPrev, z, f, i, g, o, c, h, yPred, yTarget });
    this.t++;
    return this.history[this.history.length - 1];
  }

  runAll() {
    if (!this.model) return;
    while (this.t < this.model.ys.length - 1) this.nextStep();
  }

  // ---------- DOM rendering ----------

  renderWeights() {
    const m = this.model;
    if (!m) return;
    const out = document.getElementById('lstm-weights');
    if (!out) return;
    const hLabels = Array.from({ length: m.H }, (_, i) => `h${i}`);
    const zLabels = [...hLabels, 'y'];
    const wrap = (W, b, name) => {
      let html = '';
      html += seqMatrixTable(W, {
        rowLabels: hLabels, colLabels: zLabels, digits: 2,
        caption: `${name}  (H × (H+1))`,
      });
      html += seqMatrixTable([b], { colLabels: hLabels, rowLabels: [`b_${name.slice(2)}`], digits: 3 });
      return html;
    };
    let html = `<p style="margin:6px 0;font-size:12px;color:var(--fg-muted);">Hidden size H = ${m.H}. Concat dim D = H+1 = ${m.H + 1}. Gate biases are H-dim; forget-bias initialised to +1.</p>`;
    html += '<details open><summary style="cursor:pointer;font-weight:600;">Forget gate (W_f, b_f)</summary>' + wrap(m.Wf, m.bf, 'W_f') + '</details>';
    html += '<details><summary style="cursor:pointer;font-weight:600;">Input gate (W_i, b_i)</summary>' + wrap(m.Wi, m.bi, 'W_i') + '</details>';
    html += '<details><summary style="cursor:pointer;font-weight:600;">Candidate (W_g, b_g)</summary>' + wrap(m.Wg, m.bg, 'W_g') + '</details>';
    html += '<details><summary style="cursor:pointer;font-weight:600;">Output gate (W_o, b_o)</summary>' + wrap(m.Wo, m.bo, 'W_o') + '</details>';
    html += seqMatrixTable([m.Wy], { colLabels: hLabels, rowLabels: ['W_y'], digits: 3, caption: 'W_y  (H → 1),   b_y = ' + seqFmt(m.by, 3) });
    out.innerHTML = html;
    out.parentElement.style.display = 'block';
  }

  renderHistory() {
    const m = this.model;
    const out = document.getElementById('lstm-history');
    if (!out || !m) return;
    if (this.history.length === 0) {
      out.innerHTML = '<p style="font-size:13px;color:var(--fg-muted);">Run a step to populate the table.</p>';
      out.parentElement.style.display = 'block';
      return;
    }
    const hCols = Array.from({ length: m.H }, (_, i) => `h${i}`);
    const cCols = Array.from({ length: m.H }, (_, i) => `c${i}`);
    let html = '<div class="table-container"><table><thead><tr>';
    html += '<th>t</th><th>y_t</th>';
    cCols.forEach(c => html += `<th>${c}</th>`);
    hCols.forEach(c => html += `<th>${c}</th>`);
    html += '<th>ŷ_{t+1}</th><th>target</th><th>err</th></tr></thead><tbody>';
    for (let i = 0; i < this.history.length; i++) {
      const r = this.history[i];
      const isLast = i === this.history.length - 1;
      html += `<tr${isLast ? ' style="background:var(--accent-soft);"' : ''}><th>${r.step}</th><td>${seqFmt(r.yIn, 3)}</td>`;
      for (let j = 0; j < m.H; j++) html += `<td>${seqFmt(r.c[j], 3)}</td>`;
      for (let j = 0; j < m.H; j++) html += `<td>${seqFmt(r.h[j], 3)}</td>`;
      html += `<td>${seqFmt(r.yPred, 3)}</td><td>${seqFmt(r.yTarget, 3)}</td><td>${seqFmt(r.yPred - r.yTarget, 3)}</td></tr>`;
    }
    html += '</tbody></table></div>';
    out.innerHTML = html;
    out.parentElement.style.display = 'block';
  }

  renderStep(record) {
    const m = this.model;
    const out = document.getElementById('lstm-step-detail');
    if (!out || !m || !record) return;
    const hLabels = Array.from({ length: m.H }, (_, i) => `h${i}`);
    let html = `<p style="margin:6px 0;"><strong>Step t = ${record.step}</strong>: feed y_t = <code>${seqFmt(record.yIn, 3)}</code>, predict y_{t+1}.</p>`;
    html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">';
    html += `<div>${seqMatrixTable([record.f], { colLabels: hLabels, rowLabels: ['f_t'], digits: 3, caption: 'forget gate (σ)' })}</div>`;
    html += `<div>${seqMatrixTable([record.i], { colLabels: hLabels, rowLabels: ['i_t'], digits: 3, caption: 'input gate (σ)' })}</div>`;
    html += `<div>${seqMatrixTable([record.g], { colLabels: hLabels, rowLabels: ['g_t'], digits: 3, caption: 'candidate (tanh)' })}</div>`;
    html += `<div>${seqMatrixTable([record.o], { colLabels: hLabels, rowLabels: ['o_t'], digits: 3, caption: 'output gate (σ)' })}</div>`;
    html += '</div>';
    html += seqMatrixTable([record.cPrev], { colLabels: hLabels, rowLabels: ['c_{t-1}'], digits: 3 });
    html += seqMatrixTable([record.c], { colLabels: hLabels, rowLabels: ['c_t'], digits: 3, caption: 'c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t' });
    html += seqMatrixTable([record.h], { colLabels: hLabels, rowLabels: ['h_t'], digits: 3, caption: 'h_t = o_t ⊙ tanh(c_t)' });
    html += `<p style="margin:6px 0;">ŷ_{t+1} = W_y · h_t + b_y = <strong>${seqFmt(record.yPred, 4)}</strong>  (target: ${seqFmt(record.yTarget, 4)}, error ${seqFmt(record.yPred - record.yTarget, 4)})</p>`;
    out.innerHTML = html;
    out.parentElement.style.display = 'block';
  }

  renderMetrics() {
    const m = this.model;
    if (!m || this.history.length === 0) return;
    let sse = 0;
    for (const r of this.history) sse += (r.yPred - r.yTarget) * (r.yPred - r.yTarget);
    const mse = sse / this.history.length;
    const el = document.getElementById('lstm-mse');
    if (el) el.textContent = seqFmt(mse, 4);
  }

  predictionMarkers() {
    if (!this.model || this.history.length === 0) return [];
    const m = this.model;
    return this.history.map(r => ({ x: m.xs[r.step + 1], y: r.yPred }));
  }

  // ---------- Training (numerical gradients) ----------

  _computeMSE() {
    const m = this.model;
    if (!m) return Infinity;
    const N = m.ys.length - 1;
    let sse = 0;
    let h = m.h0.slice();
    let c = m.c0.slice();
    for (let t = 0; t < N; t++) {
      const yIn = m.ys[t];
      const z = h.concat([yIn]);
      const fPre = this._gate(m.Wf, m.bf, z);
      const iPre = this._gate(m.Wi, m.bi, z);
      const gPre = this._gate(m.Wg, m.bg, z);
      const oPre = this._gate(m.Wo, m.bo, z);
      const f = fPre.map(sigmoid);
      const iG = iPre.map(sigmoid);
      const g = gPre.map(Math.tanh);
      const o = oPre.map(sigmoid);
      const cNew = new Array(m.H);
      for (let k = 0; k < m.H; k++) cNew[k] = f[k] * c[k] + iG[k] * g[k];
      const hNew = cNew.map((v, k) => o[k] * Math.tanh(v));
      let yPred = m.by;
      for (let k = 0; k < m.H; k++) yPred += m.Wy[k] * hNew[k];
      const err = yPred - m.ys[t + 1];
      sse += err * err;
      h = hNew;
      c = cNew;
    }
    return sse / N;
  }

  _eachParam(cb) {
    const m = this.model;
    const gateMats = [m.Wf, m.Wi, m.Wg, m.Wo];
    for (const W of gateMats) {
      for (let i = 0; i < W.length; i++) {
        for (let j = 0; j < W[i].length; j++) {
          cb(() => W[i][j], v => { W[i][j] = v; });
        }
      }
    }
    const gateBiases = [m.bf, m.bi, m.bg, m.bo];
    for (const b of gateBiases) {
      for (let i = 0; i < b.length; i++) {
        cb(() => b[i], v => { b[i] = v; });
      }
    }
    for (let i = 0; i < m.H; i++) {
      cb(() => m.Wy[i], v => { m.Wy[i] = v; });
    }
    cb(() => m.by, v => { m.by = v; });
  }

  train(epochs, lr, eps = 1e-4) {
    if (!this.model) return [];
    const lossHistory = [];
    for (let e = 0; e < epochs; e++) {
      const L0 = this._computeMSE();
      lossHistory.push(L0);
      const grads = [];
      this._eachParam((get, set) => {
        const v = get();
        set(v + eps);
        const Lp = this._computeMSE();
        set(v - eps);
        const Lm = this._computeMSE();
        set(v);
        grads.push({ set, v, grad: (Lp - Lm) / (2 * eps) });
      });
      for (const g of grads) g.set(g.v - lr * g.grad);
    }
    this.t = 0;
    this.history = [];
    while (this.t < this.model.ys.length - 1) this.nextStep();
    return lossHistory;
  }
}

// Expose
window.VanillaRNN = VanillaRNN;
window.LSTM = LSTM;
