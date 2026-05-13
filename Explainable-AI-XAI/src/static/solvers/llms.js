// ============================================================================
// LLMs: educational transformer + RAG implementations
// Pure-JS, no external deps. Every numeric step is intentionally laid out
// so the user can read the math, not just the result.
// ============================================================================

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

function llmTokenize(text) {
  if (!text) return [];
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s']/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 0);
}

// Seeded LCG so that "Initialize" is reproducible from a seed input.
function llmSeededRng(seed) {
  let state = (seed | 0) || 1;
  return function () {
    // Mulberry32
    state = (state + 0x6D2B79F5) | 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function llmFmt(x, digits = 3) {
  if (x === undefined || x === null || Number.isNaN(x)) return '—';
  if (Math.abs(x) < 1e-9) return '0';
  return Number(x).toFixed(digits);
}

// Render a 2-D matrix as an HTML <table>. rowLabels / colLabels optional.
function llmMatrixTable(M, opts = {}) {
  const { rowLabels = null, colLabels = null, digits = 3, caption = null, highlightRow = null } = opts;
  const rows = M.length;
  const cols = rows > 0 ? M[0].length : 0;
  let html = '<div class="table-container"><table>';
  if (caption) html += `<caption style="caption-side:top;font-weight:600;text-align:left;margin-bottom:4px;">${caption}</caption>`;
  if (colLabels) {
    html += '<thead><tr>';
    if (rowLabels) html += '<th></th>';
    for (let j = 0; j < cols; j++) {
      html += `<th>${colLabels[j] !== undefined ? colLabels[j] : j}</th>`;
    }
    html += '</tr></thead>';
  }
  html += '<tbody>';
  for (let i = 0; i < rows; i++) {
    const rowClass = (highlightRow !== null && i === highlightRow) ? ' style="background:var(--accent-soft);"' : '';
    html += `<tr${rowClass}>`;
    if (rowLabels) html += `<th>${rowLabels[i] !== undefined ? rowLabels[i] : i}</th>`;
    for (let j = 0; j < cols; j++) {
      html += `<td>${llmFmt(M[i][j], digits)}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table></div>';
  return html;
}

// Render a 1-D vector as a one-row table.
function llmVectorTable(v, labels = null, opts = {}) {
  return llmMatrixTable([v], Object.assign({ colLabels: labels }, opts));
}

// ===========================================================================
// TRANSFORMER
// ===========================================================================

class TransformerDemo {
  constructor() {
    this.reset();
  }

  reset() {
    this.state = null;          // Holds all intermediate tensors once initialized
    this.stepIndex = 0;         // Pointer to the next step to display
    this.steps = [
      'tokenize', 'embed', 'pos',
      'qkv', 'scores', 'softmax', 'output',
      'addnorm1', 'ffn', 'addnorm2',
      'logits', 'predict',
    ];
  }

  initialize(sentence, dim, seed) {
    const tokens = llmTokenize(sentence);
    if (tokens.length < 2) {
      window.showToast('Transformer needs at least two tokens. Try a longer sentence.', 'warn');
      return false;
    }
    if (tokens.length > 16) {
      window.showToast(`Sentence has ${tokens.length} tokens; using first 16 to keep matrices readable.`, 'info');
      tokens.length = 16;
    }
    // Build vocab from the unique tokens. ID = order of first appearance.
    const vocab = [];
    const vocabIdx = {};
    tokens.forEach(t => {
      if (vocabIdx[t] === undefined) {
        vocabIdx[t] = vocab.length;
        vocab.push(t);
      }
    });

    const rng = llmSeededRng(seed);
    // Embedding matrix E: (|V| x d), small-variance Gaussian via Box-Muller-ish trick
    const gauss = () => (rng() + rng() + rng() + rng() - 2) * 0.4;
    const E = Array.from({ length: vocab.length }, () =>
      Array.from({ length: dim }, () => gauss())
    );
    // Q/K/V projection matrices (d x d).
    const Wq = Array.from({ length: dim }, () => Array.from({ length: dim }, () => gauss()));
    const Wk = Array.from({ length: dim }, () => Array.from({ length: dim }, () => gauss()));
    const Wv = Array.from({ length: dim }, () => Array.from({ length: dim }, () => gauss()));
    // Position-wise feed-forward weights: d -> d_ff -> d (d_ff = 2*d for readability).
    const dff = dim * 2;
    const W1 = Array.from({ length: dim }, () => Array.from({ length: dff }, () => gauss()));
    const b1 = new Array(dff).fill(0);
    const W2 = Array.from({ length: dff }, () => Array.from({ length: dim }, () => gauss()));
    const b2 = new Array(dim).fill(0);
    // LayerNorm γ/β. Kept at the identity (γ=1, β=0) so the user can see LN
    // is "just standardisation" of each row; in real models these are learned.
    const gamma1 = new Array(dim).fill(1);
    const beta1 = new Array(dim).fill(0);
    const gamma2 = new Array(dim).fill(1);
    const beta2 = new Array(dim).fill(0);
    // Output projection back to vocabulary for next-token prediction.
    const Wout = Array.from({ length: dim }, () => Array.from({ length: vocab.length }, () => gauss()));

    this.state = {
      sentence,
      tokens,
      ids: tokens.map(t => vocabIdx[t]),
      vocab,
      vocabIdx,
      dim,
      dff,
      seed,
      E, Wq, Wk, Wv, W1, b1, W2, b2, Wout,
      gamma1, beta1, gamma2, beta2,
      // To be filled by step methods
      X: null, PE: null, Xp: null,
      Q: null, K: null, V: null,
      S: null, A: null, Y: null,
      R1: null, LN1: null, H: null, F: null, R2: null, LN2: null,
      logits: null, nextProbs: null,
    };
    this.stepIndex = 0;
    return true;
  }

  // Compute and render the next step. Returns the step name or null if done.
  nextStep() {
    if (!this.state) return null;
    if (this.stepIndex >= this.steps.length) return null;
    const name = this.steps[this.stepIndex];
    this[`_step_${name}`]();
    this.stepIndex++;
    return name;
  }

  // ----- per-step implementations -----

  _step_tokenize() {
    const s = this.state;
    const out = document.getElementById('tr-out-tokenize');
    let html = `<p style="margin:6px 0;">Vocabulary built from input: <code>${s.vocab.map((w, i) => `${i}:${w}`).join(', ')}</code></p>`;
    html += llmMatrixTable([s.ids], {
      colLabels: s.tokens,
      rowLabels: ['ID'],
      digits: 0,
      caption: `Token IDs (T = ${s.tokens.length})`,
    });
    out.innerHTML = html;
    this._show('tokenize');
  }

  _step_embed() {
    const s = this.state;
    const T = s.ids.length;
    s.X = s.ids.map(id => s.E[id].slice());
    const out = document.getElementById('tr-out-embed');
    let html = '';
    html += llmMatrixTable(s.E, {
      rowLabels: s.vocab,
      colLabels: Array.from({ length: s.dim }, (_, i) => `e${i}`),
      caption: `Embedding matrix E (|V|=${s.vocab.length}, d=${s.dim})`,
    });
    html += llmMatrixTable(s.X, {
      rowLabels: s.tokens,
      colLabels: Array.from({ length: s.dim }, (_, i) => `e${i}`),
      caption: `X = embeddings of each token (T x d) = (${T} x ${s.dim})`,
    });
    out.innerHTML = html;
    this._show('embed');
  }

  _step_pos() {
    const s = this.state;
    const T = s.tokens.length, d = s.dim;
    s.PE = Array.from({ length: T }, (_, t) =>
      Array.from({ length: d }, (_, i) => {
        const k = Math.floor(i / 2);
        const angle = t / Math.pow(10000, (2 * k) / d);
        return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
      })
    );
    s.Xp = s.X.map((row, t) => row.map((v, i) => v + s.PE[t][i]));
    const out = document.getElementById('tr-out-pos');
    let html = '';
    html += llmMatrixTable(s.PE, {
      rowLabels: s.tokens.map((tk, t) => `t=${t}`),
      colLabels: Array.from({ length: d }, (_, i) => `${i}`),
      caption: 'PE (positional encoding)',
    });
    html += llmMatrixTable(s.Xp, {
      rowLabels: s.tokens,
      colLabels: Array.from({ length: d }, (_, i) => `${i}`),
      caption: 'X\' = X + PE (positions injected)',
    });
    out.innerHTML = html;
    this._show('pos');
  }

  _step_qkv() {
    const s = this.state;
    s.Q = llmMatMul(s.Xp, s.Wq);
    s.K = llmMatMul(s.Xp, s.Wk);
    s.V = llmMatMul(s.Xp, s.Wv);
    const out = document.getElementById('tr-out-qkv');
    let html = '';
    html += '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">';
    html += `<div>${llmMatrixTable(s.Wq, { caption: 'W_Q', digits: 2 })}</div>`;
    html += `<div>${llmMatrixTable(s.Wk, { caption: 'W_K', digits: 2 })}</div>`;
    html += `<div>${llmMatrixTable(s.Wv, { caption: 'W_V', digits: 2 })}</div>`;
    html += '</div>';
    html += llmMatrixTable(s.Q, { rowLabels: s.tokens, caption: 'Q = X\' &middot; W_Q' });
    html += llmMatrixTable(s.K, { rowLabels: s.tokens, caption: 'K = X\' &middot; W_K' });
    html += llmMatrixTable(s.V, { rowLabels: s.tokens, caption: 'V = X\' &middot; W_V' });
    out.innerHTML = html;
    this._show('qkv');
  }

  _step_scores() {
    const s = this.state;
    const KT = llmTranspose(s.K);
    const scale = 1 / Math.sqrt(s.dim);
    s.S = llmMatMul(s.Q, KT).map(row => row.map(v => v * scale));
    const out = document.getElementById('tr-out-scores');
    const html = llmMatrixTable(s.S, {
      rowLabels: s.tokens.map((t, i) => `q=${t}`),
      colLabels: s.tokens.map((t, i) => `k=${t}`),
      caption: `Scaled scores S = Q K^T / sqrt(${s.dim})`,
    });
    out.innerHTML = html;
    this._show('scores');
  }

  _step_softmax() {
    const s = this.state;
    s.A = s.S.map(row => {
      const m = Math.max(...row);
      const exps = row.map(v => Math.exp(v - m));
      const sum = exps.reduce((a, b) => a + b, 0);
      return exps.map(v => v / sum);
    });
    const out = document.getElementById('tr-out-softmax');
    const html = llmMatrixTable(s.A, {
      rowLabels: s.tokens.map(t => `query: ${t}`),
      colLabels: s.tokens.map(t => `attends to: ${t}`),
      caption: 'A = row-wise softmax(S) (each row sums to 1)',
    });
    out.innerHTML = html;
    this._show('softmax');
  }

  _step_output() {
    const s = this.state;
    s.Y = llmMatMul(s.A, s.V);
    const out = document.getElementById('tr-out-output');
    let html = '';
    html += llmMatrixTable(s.Y, {
      rowLabels: s.tokens,
      colLabels: Array.from({ length: s.dim }, (_, i) => `y${i}`),
      caption: 'Y = A &middot; V (per-token contextualized output)',
    });
    // Interpret the attention: for each token, name the top key it attended to.
    let interp = '<p style="margin:8px 0;font-size:13px;"><strong>Attention summary:</strong></p><ul style="font-size:13px;margin:4px 0 0 18px;">';
    for (let i = 0; i < s.A.length; i++) {
      const row = s.A[i];
      let max = -1, mj = 0;
      for (let j = 0; j < row.length; j++) {
        if (row[j] > max) { max = row[j]; mj = j; }
      }
      interp += `<li><code>${s.tokens[i]}</code> attended most strongly to <code>${s.tokens[mj]}</code> (weight ${llmFmt(max, 3)}).</li>`;
    }
    interp += '</ul>';
    html += interp;
    out.innerHTML = html;
    this._show('output');
  }

  _step_addnorm1() {
    const s = this.state;
    s.R1 = s.Xp.map((row, i) => row.map((v, j) => v + s.Y[i][j]));
    s.LN1 = llmLayerNorm(s.R1, s.gamma1, s.beta1);
    const out = document.getElementById('tr-out-addnorm1');
    const cols = Array.from({ length: s.dim }, (_, i) => `${i}`);
    let html = '';
    html += llmMatrixTable(s.R1, {
      rowLabels: s.tokens, colLabels: cols,
      caption: 'Residual: X\' + Y (attention output added back to its input)',
    });
    html += llmMatrixTable(s.LN1, {
      rowLabels: s.tokens, colLabels: cols,
      caption: 'LN1 = LayerNorm(residual)   [each row zero-mean, unit-variance; γ=1, β=0 here]',
    });
    out.innerHTML = html;
    this._show('addnorm1');
  }

  _step_ffn() {
    const s = this.state;
    // H = ReLU(LN1 · W1 + b1)
    const Hraw = llmMatMul(s.LN1, s.W1).map(row => row.map((v, j) => v + s.b1[j]));
    s.H = Hraw.map(row => row.map(v => Math.max(0, v)));
    // F = H · W2 + b2
    s.F = llmMatMul(s.H, s.W2).map(row => row.map((v, j) => v + s.b2[j]));
    const out = document.getElementById('tr-out-ffn');
    let html = '';
    html += `<p style="margin:4px 0;font-size:13px;">A position-wise FFN: the <em>same</em> two-layer MLP is applied to every token vector independently. Here d<sub>ff</sub> = 2·d<sub>model</sub> = ${s.dff}.</p>`;
    html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">';
    html += `<div>${llmMatrixTable(s.W1, {
      rowLabels: Array.from({ length: s.dim }, (_, i) => `in${i}`),
      colLabels: Array.from({ length: s.dff }, (_, i) => `h${i}`),
      caption: `W1 (${s.dim}×${s.dff})`, digits: 2,
    })}</div>`;
    html += `<div>${llmMatrixTable(s.W2, {
      rowLabels: Array.from({ length: s.dff }, (_, i) => `h${i}`),
      colLabels: Array.from({ length: s.dim }, (_, i) => `out${i}`),
      caption: `W2 (${s.dff}×${s.dim})`, digits: 2,
    })}</div>`;
    html += '</div>';
    html += llmMatrixTable(s.H, {
      rowLabels: s.tokens,
      colLabels: Array.from({ length: s.dff }, (_, i) => `${i}`),
      caption: 'H = ReLU(LN1·W1 + b1) (hidden activations; note any ReLU-zeroed entries)',
    });
    html += llmMatrixTable(s.F, {
      rowLabels: s.tokens,
      colLabels: Array.from({ length: s.dim }, (_, i) => `${i}`),
      caption: 'F = H·W2 + b2 (back to d_model)',
    });
    out.innerHTML = html;
    this._show('ffn');
  }

  _step_addnorm2() {
    const s = this.state;
    s.R2 = s.LN1.map((row, i) => row.map((v, j) => v + s.F[i][j]));
    s.LN2 = llmLayerNorm(s.R2, s.gamma2, s.beta2);
    const out = document.getElementById('tr-out-addnorm2');
    const cols = Array.from({ length: s.dim }, (_, i) => `${i}`);
    let html = '';
    html += llmMatrixTable(s.R2, {
      rowLabels: s.tokens, colLabels: cols,
      caption: 'Residual: LN1 + F',
    });
    html += llmMatrixTable(s.LN2, {
      rowLabels: s.tokens, colLabels: cols,
      caption: 'LN2 = LayerNorm(residual)   ← block output (would feed into next block)',
    });
    out.innerHTML = html;
    this._show('addnorm2');
  }

  _step_logits() {
    const s = this.state;
    s.logits = llmMatMul(s.LN2, s.Wout);
    const out = document.getElementById('tr-out-logits');
    let html = '';
    html += `<p style="margin:4px 0;font-size:13px;">A learned linear map back to the vocabulary. Each row of W<sub>out</sub> is a per-dimension contribution to one vocab word; each column is the "direction" associated with that word.</p>`;
    html += llmMatrixTable(s.Wout, {
      rowLabels: Array.from({ length: s.dim }, (_, i) => `${i}`),
      colLabels: s.vocab,
      caption: `W_out  (${s.dim} × |V|=${s.vocab.length})`, digits: 2,
    });
    html += llmMatrixTable(s.logits, {
      rowLabels: s.tokens,
      colLabels: s.vocab,
      caption: 'Logits = LN2 · W_out  (per-token unnormalised scores over every vocab word)',
      highlightRow: s.tokens.length - 1,
    });
    out.innerHTML = html;
    this._show('logits');
  }

  _step_predict() {
    const s = this.state;
    const lastIdx = s.tokens.length - 1;
    const lastLogits = s.logits[lastIdx];
    const m = Math.max(...lastLogits);
    const exps = lastLogits.map(v => Math.exp(v - m));
    const sum = exps.reduce((a, b) => a + b, 0);
    s.nextProbs = exps.map(v => v / sum);

    const ranked = s.nextProbs
      .map((p, i) => ({ word: s.vocab[i], prob: p, idx: i }))
      .sort((a, b) => b.prob - a.prob);
    const topK = ranked.slice(0, Math.min(5, ranked.length));

    const out = document.getElementById('tr-out-predict');
    let html = '';
    html += `<p style="margin:6px 0;">Decoder-style: feed the <strong>last</strong> token&#39;s contextualized vector through the softmax over the vocabulary. The argmax becomes the predicted next word.</p>`;
    html += `<p style="margin:6px 0;font-size:13px;">Using row <code>${lastIdx}</code> of the logits (token <code>${s.tokens[lastIdx]}</code>):</p>`;
    html += llmMatrixTable([s.nextProbs], {
      colLabels: s.vocab,
      rowLabels: ['P(next)'],
      caption: 'softmax(logits[last]) — probabilities over vocabulary',
      digits: 4,
    });
    html += '<table><thead><tr><th>rank</th><th>candidate</th><th>probability</th></tr></thead><tbody>';
    topK.forEach((r, i) => {
      const style = i === 0 ? ' style="background:var(--accent-soft);"' : '';
      html += `<tr${style}><th>${i + 1}</th><td><code>${escapeHtml(r.word)}</code></td><td>${llmFmt(r.prob, 4)}</td></tr>`;
    });
    html += '</tbody></table>';
    html += `<p style="margin:8px 0;padding:8px;background:var(--warning-soft);border-radius:4px;font-size:12px;color:var(--fg);"><strong>Note:</strong> all weights are seeded random — the model is not trained, so the predicted word reflects the random projections, not language. The same architecture, after billions of gradient steps on text, is what GPT-style models use. Change the seed to see how different weight initialisations move the prediction.</p>`;
    out.innerHTML = html;
    this._show('predict');
  }

  runAll() {
    while (this.stepIndex < this.steps.length) this.nextStep();
  }

  resetUI() {
    for (const name of this.steps) {
      const el = document.getElementById(`tr-step-${name}`);
      if (el) el.style.display = 'none';
      const out = document.getElementById(`tr-out-${name}`);
      if (out) out.innerHTML = '';
    }
    const sName = document.getElementById('tr-step-name');
    if (sName) sName.textContent = 'Not Started';
  }

  _show(name) {
    const el = document.getElementById(`tr-step-${name}`);
    if (el) {
      el.style.display = 'block';
      AppUtils.kRenderInlineFormulas(el);
    }
    const sName = document.getElementById('tr-step-name');
    if (sName) sName.textContent = this._stepLabel(name);
  }

  _stepLabel(name) {
    const map = {
      tokenize: 'Tokenize',
      embed: 'Embedding',
      pos: 'Positional encoding',
      qkv: 'Q/K/V projection',
      scores: 'Attention scores',
      softmax: 'Softmax',
      output: 'Attention output',
      addnorm1: 'Add & Norm (1)',
      ffn: 'Feed-forward',
      addnorm2: 'Add & Norm (2)',
      logits: 'Output projection',
      predict: 'Next-word prediction',
    };
    return map[name] || name;
  }
}

// Matrix helpers
function llmMatMul(A, B) {
  const rA = A.length, cA = A[0].length, cB = B[0].length;
  const out = Array.from({ length: rA }, () => new Array(cB).fill(0));
  for (let i = 0; i < rA; i++) {
    for (let k = 0; k < cA; k++) {
      const a = A[i][k];
      for (let j = 0; j < cB; j++) {
        out[i][j] += a * B[k][j];
      }
    }
  }
  return out;
}

function llmTranspose(A) {
  const r = A.length, c = A[0].length;
  const out = Array.from({ length: c }, () => new Array(r).fill(0));
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[j][i] = A[i][j];
  return out;
}

// Row-wise LayerNorm:  LN(x_i) = γ ⊙ (x_i - μ_i) / sqrt(σ²_i + ε) + β
function llmLayerNorm(X, gamma, beta, eps = 1e-5) {
  return X.map(row => {
    const d = row.length;
    const mu = row.reduce((a, b) => a + b, 0) / d;
    let varSum = 0;
    for (const v of row) varSum += (v - mu) * (v - mu);
    const variance = varSum / d;
    const invStd = 1 / Math.sqrt(variance + eps);
    return row.map((x, i) => gamma[i] * (x - mu) * invStd + beta[i]);
  });
}

// ===========================================================================
// RAG
// ===========================================================================

class RAGDemo {
  constructor() {
    this.lastState = null;
  }

  run(corpusText, queryText, topK) {
    const docs = corpusText
      .split('\n')
      .map(s => s.trim())
      .filter(s => s.length > 0);
    if (docs.length === 0) {
      window.showToast('Add at least one document to the corpus.', 'warn');
      return;
    }
    if (!queryText || queryText.trim().length === 0) {
      window.showToast('Enter a query to run retrieval.', 'warn');
      return;
    }
    topK = Math.max(1, Math.min(topK || 3, docs.length));

    const docTokens = docs.map(llmTokenize);
    const queryTokens = llmTokenize(queryText);

    // Build vocabulary as the union of corpus tokens (query terms outside
    // the corpus contribute zero to similarity, so we don't include them).
    const vocab = [];
    const vocabIdx = {};
    docTokens.forEach(doc => doc.forEach(t => {
      if (vocabIdx[t] === undefined) { vocabIdx[t] = vocab.length; vocab.push(t); }
    }));

    // Document frequency, then IDF.
    const N = docs.length;
    const df = vocab.map(t => docTokens.filter(doc => doc.includes(t)).length);
    const idf = df.map(d => Math.log(N / (1 + d)) + 1);

    // TF and TF-IDF for each document.
    const tf = docTokens.map(doc => {
      const v = new Array(vocab.length).fill(0);
      doc.forEach(t => { v[vocabIdx[t]] += 1; });
      // Normalize by doc length so longer docs aren't unfairly preferred.
      const len = doc.length || 1;
      return v.map(c => c / len);
    });
    const tfidf = tf.map(row => row.map((v, j) => v * idf[j]));

    // Query TF-IDF (tokens not in vocab are dropped; this matches sklearn).
    const qTf = new Array(vocab.length).fill(0);
    queryTokens.forEach(t => {
      if (vocabIdx[t] !== undefined) qTf[vocabIdx[t]] += 1;
    });
    const qLen = queryTokens.length || 1;
    const qTfNorm = qTf.map(c => c / qLen);
    const qTfidf = qTfNorm.map((v, j) => v * idf[j]);

    const sims = tfidf.map(d => cosine(qTfidf, d));
    const ranked = sims
      .map((s, i) => ({ doc: i, sim: s }))
      .sort((a, b) => b.sim - a.sim);
    const topIdx = ranked.slice(0, topK).map(r => r.doc);

    // Stash for callers that want to inspect (e.g. tests in future).
    this.lastState = { docs, docTokens, queryTokens, vocab, df, idf, tf, tfidf, qTfidf, sims, ranked, topIdx };
    this._render(this.lastState);
    return this.lastState;
  }

  _render(s) {
    // Step 1: tokenize
    {
      let html = '<table><thead><tr><th>#</th><th>Document</th><th>Tokens</th></tr></thead><tbody>';
      s.docs.forEach((d, i) => {
        const isTop = s.topIdx ? s.topIdx.includes(i) : false;
        const rowStyle = '';
        html += `<tr${rowStyle}><th>D${i}</th><td style="text-align:left;">${escapeHtml(d)}</td>` +
                `<td style="text-align:left;">${s.docTokens[i].map(escapeHtml).join(' ')}</td></tr>`;
      });
      html += `<tr><th>q</th><td colspan="2" style="text-align:left;"><em>Query tokens:</em> ${s.queryTokens.map(escapeHtml).join(' ') || '(empty)'}</td></tr>`;
      html += '</tbody></table>';
      document.getElementById('rag-out-tokens').innerHTML = html;
      document.getElementById('rag-step-tokens').style.display = 'block';
    }

    // Step 2: IDF
    {
      const labels = s.vocab.map((t, j) => `${t} (df=${s.df[j]})`);
      const html = llmMatrixTable([s.idf], { colLabels: labels, rowLabels: ['IDF'], caption: `Vocabulary size = ${s.vocab.length}, N = ${s.docs.length}` });
      document.getElementById('rag-out-idf').innerHTML = html;
      document.getElementById('rag-step-idf').style.display = 'block';
    }

    // Step 3: TF-IDF matrix
    {
      const html = llmMatrixTable(s.tfidf, {
        rowLabels: s.docs.map((_, i) => `D${i}`),
        colLabels: s.vocab,
        caption: 'TF-IDF (rows = docs, cols = terms)',
      });
      document.getElementById('rag-out-tfidf').innerHTML = html;
      document.getElementById('rag-step-tfidf').style.display = 'block';
    }

    // Step 4: query vector
    {
      const html = llmMatrixTable([s.qTfidf], { colLabels: s.vocab, rowLabels: ['q'], caption: 'Query TF-IDF vector' });
      document.getElementById('rag-out-query').innerHTML = html;
      document.getElementById('rag-step-query').style.display = 'block';
    }

    // Step 5: cosine similarities
    {
      const rows = s.sims.map((sim, i) => [sim]);
      const html = llmMatrixTable(rows, {
        rowLabels: s.docs.map((_, i) => `D${i}`),
        colLabels: ['cos(q, d)'],
        caption: 'Cosine similarity to query',
      });
      document.getElementById('rag-out-sim').innerHTML = html;
      document.getElementById('rag-step-sim').style.display = 'block';
    }

    // Step 6: top-k
    {
      const k = s.topIdx.length;
      let html = `<p style="margin:6px 0;">Selected top-<strong>${k}</strong> documents:</p>`;
      html += '<table><thead><tr><th>rank</th><th>doc</th><th>similarity</th><th>text</th></tr></thead><tbody>';
      s.ranked.slice(0, k).forEach((r, rank) => {
        html += `<tr style="background:var(--accent-soft);"><th>${rank + 1}</th><th>D${r.doc}</th><td>${llmFmt(r.sim, 4)}</td><td style="text-align:left;">${escapeHtml(s.docs[r.doc])}</td></tr>`;
      });
      html += '</tbody></table>';
      document.getElementById('rag-out-topk').innerHTML = html;
      document.getElementById('rag-step-topk').style.display = 'block';
    }

    // Step 7: augmented prompt
    {
      const context = s.topIdx
        .map((i, rank) => `[${rank + 1}] ${s.docs[i]}`)
        .join('\n');
      const queryText = s.queryTokens.join(' ');
      const prompt = `You are a helpful assistant. Use ONLY the context below to answer.\n\nContext:\n${context}\n\nQuestion: ${queryText}\nAnswer:`;
      const html = `<pre style="background:var(--code-bg);color:var(--code-fg);padding:12px;border-radius:6px;white-space:pre-wrap;font-size:12px;margin:0;">${escapeHtml(prompt)}</pre>`;
      document.getElementById('rag-out-prompt').innerHTML = html;
      document.getElementById('rag-step-prompt').style.display = 'block';
    }

    // Render KaTeX formulas in any newly-visible step
    ['idf', 'tfidf', 'sim'].forEach(name => {
      const el = document.getElementById(`rag-step-${name}`);
      if (el) AppUtils.kRenderInlineFormulas(el);
    });
  }

  resetUI() {
    ['tokens', 'idf', 'tfidf', 'query', 'sim', 'topk', 'prompt'].forEach(name => {
      const el = document.getElementById(`rag-step-${name}`);
      if (el) el.style.display = 'none';
      const out = document.getElementById(`rag-out-${name}`);
      if (out) out.innerHTML = '';
    });
  }
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / Math.sqrt(na * nb);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// Expose
window.TransformerDemo = TransformerDemo;
window.RAGDemo = RAGDemo;
