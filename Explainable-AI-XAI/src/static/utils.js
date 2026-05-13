// ============================================================================
// UTILITIES
// ============================================================================

const DEBUG = false;
function dlog(...args) { if (DEBUG) console.log(...args); }
function dwarn(...args) { if (DEBUG) console.warn(...args); }
function derror(...args) { if (DEBUG) console.error(...args); }

// Toast notifications (replaces silent throws / alerts)
function showToast(message, level = 'info', duration = 3500) {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    document.body.appendChild(container);
  }
  const toast = document.createElement('div');
  toast.className = `toast toast-${level}`;
  toast.textContent = message;
  container.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('toast-show'));
  setTimeout(() => {
    toast.classList.remove('toast-show');
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// KaTeX rendering utilities
const katexQueue = [];

function kRender(el, tex, display = false) {
  if (!el) return;
  if (window.katex) {
    window.katex.render(tex, el, {displayMode: display, throwOnError: false});
  } else {
    katexQueue.push([el, tex, display]);
  }
}

function kFlush() {
  if (!window.katex) return;
  while (katexQueue.length) {
    const [el, tex, display] = katexQueue.shift();
    window.katex.render(tex, el, {displayMode: display, throwOnError: false});
  }
}

function kRenderInlineFormulas(container) {
  if (!window.katex || !container) return;
  const formulas = container.querySelectorAll('.formula-inline[data-formula]');
  formulas.forEach(el => {
    const formula = el.getAttribute('data-formula');
    window.katex.render(formula, el, {displayMode: false, throwOnError: false});
  });
}

// Math utilities
function calculateMSE(points, m, b) {
  if (!points.length) return 0;
  return points.reduce((sum, p) => sum + Math.pow(p.y - (m * p.x + b), 2), 0) / points.length;
}

function calculateR2(points, m, b) {
  const n = points.length;
  if (!n) return 0;
  const meanY = points.reduce((s, p) => s + p.y, 0) / n;
  let ssRes = 0, ssTot = 0;
  for (const p of points) {
    const pred = m * p.x + b;
    ssRes += Math.pow(p.y - pred, 2);
    ssTot += Math.pow(p.y - meanY, 2);
  }
  return ssTot > 0 ? 1 - ssRes / ssTot : 0;
}

// Domain utilities
const domain = {xmin: 0, xmax: 10, ymin: 0, ymax: 10};
const margin = 50;

function dataToPx(x, y, canvas) {
  const w = canvas.width - 2 * margin, h = canvas.height - 2 * margin;
  const px = margin + (x - domain.xmin) / (domain.xmax - domain.xmin) * w;
  const py = margin + h - (y - domain.ymin) / (domain.ymax - domain.ymin) * h;
  return [px, py];
}

function pxToData(px, py, canvas) {
  const w = canvas.width - 2 * margin, h = canvas.height - 2 * margin;
  const nx = Math.max(0, Math.min(1, (px - margin) / w));
  const ny = Math.max(0, Math.min(1, 1 - (py - margin) / h));
  const x = domain.xmin + nx * (domain.xmax - domain.xmin);
  const y = domain.ymin + ny * (domain.ymax - domain.ymin);
  return [x, y];
}

// Export for global access
window.AppUtils = {
  kRender, kFlush, kRenderInlineFormulas, calculateMSE, calculateR2,
  dataToPx, pxToData, domain, margin, showToast
};
window.showToast = showToast;

// ============================================================================
// CANVAS HANDLER
// ============================================================================

class CanvasHandler {
  constructor() {
    this.canvases = {
      ols: document.getElementById('plot-ols'),
      gd: document.getElementById('plot-gd'),
      manual: document.getElementById('plot-manual'),
      logistic: document.getElementById('plot-logistic'),
      dt: document.getElementById('plot-dt'),
      rf: document.getElementById('plot-rf'),
      xgb: document.getElementById('plot-xgb'),
      nn: document.getElementById('plot-nn'),
      rnn: document.getElementById('plot-rnn'),
      lstm: document.getElementById('plot-lstm'),
      clustering: document.getElementById('plot-clustering')
    };
    this.contexts = {};
    Object.keys(this.canvases).forEach(key => {
      if (this.canvases[key]) {
        this.contexts[key] = this.canvases[key].getContext('2d');
      }
    });
    this.lossCanvas = document.getElementById('loss-plot');
    if (this.lossCanvas) {
      this.lossCtx = this.lossCanvas.getContext('2d');
    }
    this.dragging = -1;
    this.setupEvents();
    this.setupResponsiveCanvases();
  }

  // Set backing-store size on every visible canvas so it tracks CSS layout size,
  // then redraw on resize (debounced via rAF).
  setupResponsiveCanvases() {
    const allCanvases = Array.from(document.querySelectorAll('canvas'));
    const dpr = window.devicePixelRatio || 1;

    const sizeOne = (canvas) => {
      const rect = canvas.getBoundingClientRect();
      const w = Math.max(50, Math.round(rect.width));
      const h = Math.max(50, Math.round(rect.height));
      // Set the backing store; we don't scale the context — drawing code uses
      // canvas.width / canvas.height directly so the existing math stays valid.
      // DPR is intentionally ignored to keep the existing margin/font sizes consistent.
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
        return true;
      }
      return false;
    };

    // Initial size pass
    allCanvases.forEach(sizeOne);

    if (typeof ResizeObserver === 'undefined') return;

    let pending = null;
    const requestRedraw = () => {
      if (pending) return;
      pending = requestAnimationFrame(() => {
        pending = null;
        if (window.App && typeof window.App.redraw === 'function') {
          window.App.redraw();
        }
      });
    };

    const ro = new ResizeObserver(entries => {
      let changed = false;
      for (const entry of entries) {
        if (sizeOne(entry.target)) changed = true;
      }
      if (changed) requestRedraw();
    });
    allCanvases.forEach(c => ro.observe(c));
  }

  draw(activeTab, points, models, showResiduals) {
    const canvas = this.canvases[activeTab];
    if (!canvas) return; // Tab not yet initialized
    const ctx = this.contexts[activeTab];

    if (activeTab === 'logistic') {
      this.drawLogisticCanvas(ctx, canvas, points, models[activeTab], showResiduals);
    } else {
      this.drawCanvas(ctx, canvas, points, models[activeTab], showResiduals);
    }

    const countEl = document.getElementById('point-count');
    if (countEl) countEl.textContent = points.length;
  }

  drawCanvas(ctx, canvas, points, model, showResiduals) {
    // Clear
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Grid
    this.drawGrid(ctx, canvas);
    // Axes  
    this.drawAxes(ctx, canvas);
    // Fit + residuals
    if (model.fitted) {
      this.drawFit(ctx, canvas, model, points, showResiduals);
    }
    // Points
    this.drawPoints(ctx, canvas, points);
  }

  drawGrid(ctx, canvas) {
    ctx.strokeStyle = '#f1f5f9';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = AppUtils.margin + i * (canvas.width - 2 * AppUtils.margin) / 10;
      const y = AppUtils.margin + i * (canvas.height - 2 * AppUtils.margin) / 10;
      ctx.beginPath();
      ctx.moveTo(x, AppUtils.margin); 
      ctx.lineTo(x, canvas.height - AppUtils.margin);
      ctx.moveTo(AppUtils.margin, y); 
      ctx.lineTo(canvas.width - AppUtils.margin, y);
      ctx.stroke();
    }
  }

  drawAxes(ctx, canvas) {
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 2;
    ctx.strokeRect(AppUtils.margin, AppUtils.margin, 
                   canvas.width - 2 * AppUtils.margin, 
                   canvas.height - 2 * AppUtils.margin);

    // Labels
    ctx.fillStyle = '#374151';
    ctx.font = '12px system-ui';
    ctx.textAlign = 'center';
    for (let i = 0; i <= 10; i++) {
      const xVal = AppUtils.domain.xmin + i;
      const [px] = AppUtils.dataToPx(xVal, AppUtils.domain.ymin, canvas);
      ctx.fillText(xVal.toFixed(0), px, canvas.height - AppUtils.margin + 20);
    }
    ctx.textAlign = 'right';
    for (let i = 0; i <= 10; i++) {
      const yVal = AppUtils.domain.ymin + i;
      const [, py] = AppUtils.dataToPx(AppUtils.domain.xmin, yVal, canvas);
      ctx.fillText(yVal.toFixed(0), AppUtils.margin - 10, py + 4);
    }
  }

  drawFit(ctx, canvas, model, points, showResiduals) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(AppUtils.margin, AppUtils.margin, 
             canvas.width - 2 * AppUtils.margin, 
             canvas.height - 2 * AppUtils.margin);
    ctx.clip();

    const y1 = model.m * AppUtils.domain.xmin + model.b;
    const y2 = model.m * AppUtils.domain.xmax + model.b;
    const [x1, py1] = AppUtils.dataToPx(AppUtils.domain.xmin, y1, canvas);
    const [x2, py2] = AppUtils.dataToPx(AppUtils.domain.xmax, y2, canvas);
    
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x1, py1);
    ctx.lineTo(x2, py2);
    ctx.stroke();

    if (showResiduals) {
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      for (const p of points) {
        const pred = model.m * p.x + model.b;
        const [px, py] = AppUtils.dataToPx(p.x, p.y, canvas);
        const [, pyPred] = AppUtils.dataToPx(p.x, pred, canvas);
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px, pyPred);
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  drawPoints(ctx, canvas, points) {
    let anyTest = false;
    for (const p of points) {
      const [px, py] = AppUtils.dataToPx(p.x, p.y, canvas);
      if (p.isTest) {
        anyTest = true;
        // Test points: hollow ring with thicker stroke
        ctx.beginPath();
        ctx.arc(px, py, 7, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(255,255,255,0.85)';
        ctx.fill();
        ctx.strokeStyle = '#ea580c';
        ctx.lineWidth = 2.5;
        ctx.stroke();
      } else {
        ctx.fillStyle = '#1f2937';
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
    if (anyTest) this._drawSplitLegend(ctx, canvas, 'regression');
  }

  drawLogisticCanvas(ctx, canvas, points, model, showResiduals) {
    // Clear
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Grid
    this.drawGrid(ctx, canvas);
    // Axes
    this.drawAxes(ctx, canvas);

    // Decision boundary
    if (model.fitted) {
      this.drawDecisionBoundary(ctx, canvas, model);
    }

    // Points with class colors
    this.drawClassificationPoints(ctx, canvas, points);
  }

  drawDecisionBoundary(ctx, canvas, model) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(AppUtils.margin, AppUtils.margin,
             canvas.width - 2 * AppUtils.margin,
             canvas.height - 2 * AppUtils.margin);
    ctx.clip();

    // Decision boundary is where mx + b = 0 (sigmoid = 0.5)
    // So x = -b/m
    if (Math.abs(model.m) > 1e-6) {
      const xBoundary = -model.b / model.m;
      if (xBoundary >= AppUtils.domain.xmin && xBoundary <= AppUtils.domain.xmax) {
        const [px] = AppUtils.dataToPx(xBoundary, AppUtils.domain.ymin, canvas);
        ctx.strokeStyle = '#8b5cf6';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(px, AppUtils.margin);
        ctx.lineTo(px, canvas.height - AppUtils.margin);
        ctx.stroke();
      }
    }

    ctx.restore();
  }

  drawClassificationPoints(ctx, canvas, points) {
    let anyTest = false;
    for (const p of points) {
      const [px, py] = AppUtils.dataToPx(p.x, p.y, canvas);
      const classColor = p.label === 1 ? '#3b82f6' : '#ef4444';
      if (p.isTest) {
        anyTest = true;
        // Hollow ring in the class color
        ctx.beginPath();
        ctx.arc(px, py, 7, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(255,255,255,0.85)';
        ctx.fill();
        ctx.strokeStyle = classColor;
        ctx.lineWidth = 2.5;
        ctx.stroke();
      } else {
        ctx.fillStyle = classColor;
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
    if (anyTest) this._drawSplitLegend(ctx, canvas, 'classification');
  }

  // Nonlinear Models Drawing
  drawNL(model, points, models, taskType) {
    const canvas = this.canvases[model];
    if (!canvas) return;
    const ctx = this.contexts[model];

    // Clear
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Grid
    this.drawGrid(ctx, canvas);
    // Axes
    this.drawAxes(ctx, canvas);

    // Sequence models (RNN / LSTM) have their own prediction layout: open-circle
    // markers at each (x_{t+1}, ŷ_{t+1}) connected by a line.
    if (model === 'rnn' || model === 'lstm') {
      const m = models[model];
      if (m && m.fitted && Array.isArray(m.predictions) && m.predictions.length > 0) {
        this.drawSequencePredictions(ctx, canvas, points, m.predictions);
      }
      this.drawPoints(ctx, canvas, points);
      return;
    }

    // Draw model predictions/boundaries
    if (models[model] && models[model].fitted) {
      if (taskType === 'classification') {
        this.drawNLDecisionBoundary(ctx, canvas, models[model], model);
      } else {
        this.drawNLPredictions(ctx, canvas, models[model], model);
      }
    }

    // Points
    if (taskType === 'classification') {
      this.drawClassificationPoints(ctx, canvas, points);
    } else {
      this.drawPoints(ctx, canvas, points);
    }
  }

  // Draw a small in-canvas legend showing train vs test point styles in the
  // upper-right corner. Called only when at least one test point exists.
  _drawSplitLegend(ctx, canvas, mode) {
    const padX = 8, padY = 6, rowH = 16, dotR = 5;
    const lines = [
      { kind: 'train', label: 'train' },
      { kind: 'test', label: 'test (held out)' },
    ];
    ctx.save();
    ctx.font = '11px system-ui, sans-serif';
    const w = Math.max(...lines.map(l => ctx.measureText(l.label).width)) + 28 + padX * 2;
    const h = rowH * lines.length + padY * 2;
    const x = canvas.width - AppUtils.margin - w - 4;
    const y = AppUtils.margin + 4;

    // Background
    ctx.fillStyle = 'rgba(255,255,255,0.92)';
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.fill();
    ctx.stroke();

    for (let i = 0; i < lines.length; i++) {
      const cy = y + padY + rowH * i + rowH / 2;
      const cx = x + padX + dotR;
      const line = lines[i];
      ctx.beginPath();
      ctx.arc(cx, cy, dotR, 0, 2 * Math.PI);
      if (line.kind === 'train') {
        ctx.fillStyle = (mode === 'classification') ? '#1f2937' : '#1f2937';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      } else {
        ctx.fillStyle = 'rgba(255,255,255,0.85)';
        ctx.fill();
        ctx.strokeStyle = '#ea580c';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      ctx.fillStyle = '#0f172a';
      ctx.textBaseline = 'middle';
      ctx.fillText(line.label, cx + dotR + 6, cy);
    }
    ctx.restore();
  }

  drawSequencePredictions(ctx, canvas, points, predictions) {
    if (!predictions || predictions.length === 0) return;
    ctx.save();
    ctx.beginPath();
    ctx.rect(AppUtils.margin, AppUtils.margin,
             canvas.width - 2 * AppUtils.margin,
             canvas.height - 2 * AppUtils.margin);
    ctx.clip();

    // Connecting line (predicted trajectory)
    ctx.strokeStyle = '#16a34a';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < predictions.length; i++) {
      const [px, py] = AppUtils.dataToPx(predictions[i].x, predictions[i].y, canvas);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Open-circle markers at each predicted point
    for (const p of predictions) {
      const [px, py] = AppUtils.dataToPx(p.x, p.y, canvas);
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      ctx.fill();
      ctx.strokeStyle = '#16a34a';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    ctx.restore();
  }

  drawNLDecisionBoundary(ctx, canvas, model, modelType) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(AppUtils.margin, AppUtils.margin,
             canvas.width - 2 * AppUtils.margin,
             canvas.height - 2 * AppUtils.margin);
    ctx.clip();

    // Sample many x points and determine class
    const steps = 200;
    const dx = (AppUtils.domain.xmax - AppUtils.domain.xmin) / steps;

    // Get the predictor based on model type
    let predictor;
    if (modelType === 'dt') {
      const dt = new DecisionTree();
      predictor = (x) => dt.predict({x: x}, model.tree);
    } else if (modelType === 'rf') {
      const rf = new RandomForest();
      predictor = (x) => rf.predict({x: x}, model.trees, 'classification');
    } else if (modelType === 'xgb') {
      const xgb = new XGBoost();
      predictor = (x) => xgb.predict({x: x}, model, 'classification');
    } else if (modelType === 'nn') {
      const nn = new NeuralNetwork();
      predictor = (x) => nn.predict({x: x}, model, 'classification');
    }

    // Draw decision regions by sampling and finding boundaries
    for (let i = 0; i < steps - 1; i++) {
      const x1 = AppUtils.domain.xmin + i * dx;
      const x2 = AppUtils.domain.xmin + (i + 1) * dx;

      const class1 = predictor(x1);
      const class2 = predictor(x2);

      // If classes differ, there's a decision boundary between x1 and x2
      if (class1 !== class2) {
        const xBoundary = (x1 + x2) / 2;
        const [px] = AppUtils.dataToPx(xBoundary, AppUtils.domain.ymin, canvas);

        ctx.strokeStyle = '#8b5cf6';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(px, AppUtils.margin);
        ctx.lineTo(px, canvas.height - AppUtils.margin);
        ctx.stroke();
      }
    }

    ctx.restore();
  }

  drawNLPredictions(ctx, canvas, model, modelType) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(AppUtils.margin, AppUtils.margin,
             canvas.width - 2 * AppUtils.margin,
             canvas.height - 2 * AppUtils.margin);
    ctx.clip();

    // Get the predictor based on model type
    let predictor;
    if (modelType === 'dt') {
      const dt = new DecisionTree();
      predictor = (x) => dt.predict({x: x}, model.tree);
    } else if (modelType === 'rf') {
      const rf = new RandomForest();
      predictor = (x) => rf.predict({x: x}, model.trees, 'regression');
    } else if (modelType === 'xgb') {
      const xgb = new XGBoost();
      predictor = (x) => xgb.predictRaw({x: x}, model);
    } else if (modelType === 'nn') {
      const nn = new NeuralNetwork();
      predictor = (x) => nn.predictRaw({x: x}, model);
    }

    // Draw the prediction curve
    const steps = 200;
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 3;
    ctx.beginPath();

    for (let i = 0; i <= steps; i++) {
      const x = AppUtils.domain.xmin + (i / steps) * (AppUtils.domain.xmax - AppUtils.domain.xmin);
      const y = predictor(x);
      const [px, py] = AppUtils.dataToPx(x, y, canvas);

      if (i === 0) {
        ctx.moveTo(px, py);
      } else {
        ctx.lineTo(px, py);
      }
    }
    ctx.stroke();

    ctx.restore();
  }

  drawLossPlot(lossHistory, canvasId = 'loss-plot') {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, w, h);
    if (lossHistory.length < 2) return;

    const marginL = 30, marginT = 20, marginR = 10, marginB = 30;
    const minLoss = Math.min(...lossHistory);
    const maxLoss = Math.max(...lossHistory);
    const range = maxLoss - minLoss || 1;

    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.strokeRect(marginL, marginT, w - marginL - marginR, h - marginT - marginB);

    ctx.strokeStyle = '#dc2626';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < lossHistory.length; i++) {
      const x = marginL + i * (w - marginL - marginR) / (lossHistory.length - 1);
      const y = marginT + (1 - (lossHistory[i] - minLoss) / range) * (h - marginT - marginB);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  drawStatisticsPlot(canvasId, data, plotType) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;

    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, w, h);

    if (plotType === 'correlation') {
      this.drawCorrelationPlot(ctx, canvas, data);
    } else if (plotType === 'ttest') {
      this.drawTTestPlot(ctx, canvas, data);
    } else if (plotType === 'anova') {
      this.drawANOVAPlot(ctx, canvas, data);
    }
  }

  drawCorrelationPlot(ctx, canvas, data) {
    const { group1, group2 } = data;
    const n = Math.min(group1.length, group2.length);

    const margin = 50;
    const w = canvas.width - 2 * margin;
    const h = canvas.height - 2 * margin;

    // Use 0-10 scale for both axes
    const minVal = 0;
    const maxVal = 10;

    // Draw axes
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 2;
    ctx.strokeRect(margin, margin, w, h);

    // Draw x-axis labels
    ctx.fillStyle = '#374151';
    ctx.font = '12px system-ui';
    ctx.textAlign = 'center';
    for (let i = 0; i <= 10; i++) {
      const px = margin + (i / 10) * w;
      ctx.fillText(i.toString(), px, canvas.height - margin + 20);
    }

    // Draw y-axis labels
    ctx.textAlign = 'right';
    for (let i = 0; i <= 10; i++) {
      const py = margin + h - (i / 10) * h;
      ctx.fillText(i.toString(), margin - 10, py + 4);
    }

    // Draw points
    if (n > 0) {
      ctx.fillStyle = '#3b82f6';
      for (let i = 0; i < n; i++) {
        const px = margin + (group1[i] / maxVal) * w;
        const py = margin + h - (group2[i] / maxVal) * h;
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, 2 * Math.PI);
        ctx.fill();
      }
    }

    // Axis labels
    ctx.fillStyle = '#374151';
    ctx.font = '12px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('Variable 1', canvas.width / 2, canvas.height - 10);
    ctx.save();
    ctx.translate(15, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Variable 2', 0, 0);
    ctx.restore();
  }

  drawTTestPlot(ctx, canvas, data) {
    const { group1, group2, mean1, mean2 } = data;
    const margin = 50;
    const w = canvas.width - 2 * margin;
    const h = canvas.height - 2 * margin;

    // Use 0-10 scale
    const minVal = 0;
    const maxVal = 10;
    const range = 10;

    // Draw boundary regions with light backgrounds
    ctx.fillStyle = '#fee2e2'; // Light red for group 1
    ctx.fillRect(margin, margin, w / 2, h);
    ctx.fillStyle = '#dbeafe'; // Light blue for group 2
    ctx.fillRect(margin + w / 2, margin, w / 2, h);

    // Draw center divider
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(margin + w / 2, margin);
    ctx.lineTo(margin + w / 2, margin + h);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw axes
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 2;
    ctx.strokeRect(margin, margin, w, h);

    // Draw y-axis labels
    ctx.fillStyle = '#374151';
    ctx.font = '12px system-ui';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 10; i++) {
      const py = margin + h - (i / 10) * h;
      ctx.fillText(i.toString(), margin - 10, py + 4);
    }

    const drawGroup = (group, xPos, color, label) => {
      ctx.fillStyle = color;
      group.forEach((val, idx) => {
        const py = margin + h - (val / maxVal) * h;
        const px = xPos + ((idx % 10) - 4.5) * 5; // Spread horizontally
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });

      // Label
      ctx.fillStyle = '#374151';
      ctx.font = '14px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText(label, xPos, margin - 20);
    };

    drawGroup(group1, margin + w * 0.25, '#dc2626', 'Group 1 (Left)');
    drawGroup(group2, margin + w * 0.75, '#2563eb', 'Group 2 (Right)');

    // Draw means if data exists
    if (group1.length > 0) {
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 3;
      const mean1Py = margin + h - (mean1 / maxVal) * h;
      ctx.beginPath();
      ctx.moveTo(margin + 10, mean1Py);
      ctx.lineTo(margin + w / 2 - 10, mean1Py);
      ctx.stroke();
    }

    if (group2.length > 0) {
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 3;
      const mean2Py = margin + h - (mean2 / maxVal) * h;
      ctx.beginPath();
      ctx.moveTo(margin + w / 2 + 10, mean2Py);
      ctx.lineTo(margin + w - 10, mean2Py);
      ctx.stroke();
    }
  }

  drawANOVAPlot(ctx, canvas, data) {
    const { groups, groupMeans } = data;
    const margin = 50;
    const w = canvas.width - 2 * margin;
    const h = canvas.height - 2 * margin;

    // Use 0-10 scale
    const minVal = 0;
    const maxVal = 10;
    const range = 10;

    // Draw each group with background colors (show all 5 regions)
    const colors = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6'];
    const bgColors = ['#fee2e2', '#dbeafe', '#dcfce7', '#fed7aa', '#ede9fe'];
    const numRegions = 5;
    const regionWidth = w / numRegions;

    // Draw background regions for all 5 groups
    for (let i = 0; i < numRegions; i++) {
      ctx.fillStyle = bgColors[i % bgColors.length];
      ctx.fillRect(margin + i * regionWidth, margin, regionWidth, h);
    }

    // Draw dividers between regions
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    for (let i = 1; i < numRegions; i++) {
      ctx.beginPath();
      ctx.moveTo(margin + i * regionWidth, margin);
      ctx.lineTo(margin + i * regionWidth, margin + h);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Draw axes
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 2;
    ctx.strokeRect(margin, margin, w, h);

    // Draw y-axis labels
    ctx.fillStyle = '#374151';
    ctx.font = '12px system-ui';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 10; i++) {
      const py = margin + h - (i / 10) * h;
      ctx.fillText(i.toString(), margin - 10, py + 4);
    }

    // Draw group labels for all 5 regions
    ctx.fillStyle = '#374151';
    ctx.font = '12px system-ui';
    ctx.textAlign = 'center';
    for (let i = 0; i < numRegions; i++) {
      const xCenter = margin + regionWidth * (i + 0.5);
      ctx.fillText(`G${i + 1}`, xCenter, canvas.height - margin + 20);
    }

    // Draw points for each group that has data
    groups.forEach((group, i) => {
      const xCenter = margin + regionWidth * (i + 0.5);
      ctx.fillStyle = colors[i % colors.length];

      // Draw points with deterministic positioning
      group.forEach((val, j) => {
        const py = margin + h - (val / maxVal) * h;
        // Use deterministic offset based on index
        const offset = ((j % 10) - 4.5) * (regionWidth * 0.08);
        const px = xCenter + offset;
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });

      // Draw mean line
      if (groupMeans[i] !== undefined) {
        const meanY = margin + h - (groupMeans[i] / maxVal) * h;
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(margin + i * regionWidth + 10, meanY);
        ctx.lineTo(margin + (i + 1) * regionWidth - 10, meanY);
        ctx.stroke();
      }
    });
  }

  drawClustering(points, assignments, centers) {
    const canvas = this.canvases.clustering;
    if (!canvas) return;
    const ctx = this.contexts.clustering;

    // Clear
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Grid and axes
    this.drawGrid(ctx, canvas);
    this.drawAxes(ctx, canvas);

    // Colors for clusters
    const colors = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316', '#a855f7', '#84cc16'];

    // Draw points colored by cluster
    points.forEach((point, idx) => {
      const [px, py] = AppUtils.dataToPx(point.x, point.y, canvas);
      const clusterIdx = assignments && assignments[idx] !== undefined ? assignments[idx] : -1;

      if (clusterIdx >= 0) {
        ctx.fillStyle = colors[clusterIdx % colors.length];
      } else {
        ctx.fillStyle = '#1f2937';
      }

      ctx.beginPath();
      ctx.arc(px, py, 6, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    // Draw centroids (for K-means)
    if (centers && centers.length > 0) {
      centers.forEach((center, idx) => {
        const [px, py] = AppUtils.dataToPx(center.x, center.y, canvas);

        // Draw centroid as a larger X marker
        ctx.strokeStyle = colors[idx % colors.length];
        ctx.lineWidth = 4;

        // Draw X
        const size = 10;
        ctx.beginPath();
        ctx.moveTo(px - size, py - size);
        ctx.lineTo(px + size, py + size);
        ctx.moveTo(px + size, py - size);
        ctx.lineTo(px - size, py + size);
        ctx.stroke();

        // Draw outline
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(px, py, size + 2, 0, 2 * Math.PI);
        ctx.stroke();
      });
    }

    // Update point count
    const countEl = document.getElementById('clustering-point-count');
    if (countEl) countEl.textContent = points.length;
  }

  setupEvents() {
    dlog('Setting up canvas events...', this.canvases);
    Object.values(this.canvases).forEach(canvas => {
      if (!canvas) {
        dwarn('Canvas element not found!');
        return;
      }
      dlog('Attaching events to canvas:', canvas.id);
      canvas.addEventListener('contextmenu', e => e.preventDefault());
      canvas.addEventListener('mousedown', e => this.onMouseDown(e, canvas));
      canvas.addEventListener('mousemove', e => this.onMouseMove(e, canvas));
      canvas.addEventListener('mouseup', () => this.dragging = -1);
      canvas.addEventListener('mouseleave', () => this.dragging = -1);

      // Touch support: emulate a left-click drag.
      const touchEvt = (touch, button = 0) => ({
        clientX: touch.clientX,
        clientY: touch.clientY,
        button,
        shiftKey: false,
        preventDefault() {},
      });
      canvas.addEventListener('touchstart', e => {
        if (e.touches.length !== 1) return;
        e.preventDefault();
        this.onMouseDown(touchEvt(e.touches[0]), canvas);
      }, { passive: false });
      canvas.addEventListener('touchmove', e => {
        if (e.touches.length !== 1) return;
        e.preventDefault();
        this.onMouseMove(touchEvt(e.touches[0]), canvas);
      }, { passive: false });
      canvas.addEventListener('touchend', () => { this.dragging = -1; });
      canvas.addEventListener('touchcancel', () => { this.dragging = -1; });
    });

    // Statistics canvas event handlers
    const corrCanvas = document.getElementById('plot-correlation');
    if (corrCanvas) {
      corrCanvas.addEventListener('click', e => this.onStatisticsClick(e, corrCanvas, 'correlation'));
      corrCanvas.addEventListener('contextmenu', e => {
        e.preventDefault();
        this.onStatisticsRightClick('correlation');
      });
    }

    const ttestCanvas = document.getElementById('plot-ttest');
    if (ttestCanvas) {
      ttestCanvas.addEventListener('click', e => this.onStatisticsClick(e, ttestCanvas, 'ttest'));
      ttestCanvas.addEventListener('contextmenu', e => {
        e.preventDefault();
        this.onStatisticsRightClick('ttest');
      });
    }

    const anovaCanvas = document.getElementById('plot-anova');
    if (anovaCanvas) {
      anovaCanvas.addEventListener('click', e => this.onStatisticsClick(e, anovaCanvas, 'anova'));
      anovaCanvas.addEventListener('contextmenu', e => {
        e.preventDefault();
        this.onStatisticsRightClick('anova');
      });
    }
  }

  onStatisticsClick(e, canvas, type) {
    const [px, py] = this.getMousePos(e, canvas);
    const w = canvas.width, h = canvas.height;
    const margin = 50;

    if (type === 'correlation') {
      // Map click to 0-10 data coordinates for both axes
      const x = ((px - margin) / (w - 2 * margin)) * 10;
      const y = ((h - margin - py) / (h - 2 * margin)) * 10;

      if (x >= 0 && x <= 10 && y >= 0 && y <= 10) {
        window.App.statData.correlation.push({ x, y });
        document.getElementById('corr-point-count').textContent = window.App.statData.correlation.length;
        const group1 = window.App.statData.correlation.map(p => p.x);
        const group2 = window.App.statData.correlation.map(p => p.y);
        this.drawStatisticsPlot('plot-correlation', { group1, group2 }, 'correlation');
      }
    } else if (type === 'ttest') {
      // Determine which group based on x position (left or right half)
      const halfWidth = (w - 2 * margin) / 2;
      const clickX = px - margin;
      const activeGroup = clickX < halfWidth ? 0 : 1;

      // Update active group selection
      window.App.statData.ttest.activeGroup = activeGroup;
      window.App.setTTestGroup(activeGroup);

      // Map to 0-10 coordinate space
      const value = ((h - margin - py) / (h - 2 * margin)) * 10;

      if (value >= 0 && value <= 10 && clickX >= 0 && clickX < (w - 2 * margin)) {
        if (activeGroup === 0) {
          window.App.statData.ttest.group1.push(value);
          document.getElementById('ttest-n1').textContent = window.App.statData.ttest.group1.length;
        } else {
          window.App.statData.ttest.group2.push(value);
          document.getElementById('ttest-n2').textContent = window.App.statData.ttest.group2.length;
        }

        const mean1 = window.App.statData.ttest.group1.length > 0 ?
          window.App.statData.ttest.group1.reduce((s,v) => s+v, 0) / window.App.statData.ttest.group1.length : 0;
        const mean2 = window.App.statData.ttest.group2.length > 0 ?
          window.App.statData.ttest.group2.reduce((s,v) => s+v, 0) / window.App.statData.ttest.group2.length : 0;
        this.drawStatisticsPlot('plot-ttest', {...window.App.statData.ttest, mean1, mean2 }, 'ttest');
      }
    } else if (type === 'anova') {
      // Determine which group based on x position (divide into 5 regions)
      const clickX = px - margin;
      const regionWidth = (w - 2 * margin) / 5;
      const activeGroup = Math.floor(clickX / regionWidth);

      if (activeGroup >= 0 && activeGroup < 5) {
        // Update active group selection
        window.App.statData.anova.activeGroup = activeGroup;
        window.App.setAnovaGroup(activeGroup);

        // Map to 0-10 coordinate space
        const value = ((h - margin - py) / (h - 2 * margin)) * 10;

        if (value >= 0 && value <= 10 && clickX >= 0 && clickX < (w - 2 * margin)) {
          window.App.statData.anova.groups[activeGroup].push(value);
          for (let i = 1; i <= 5; i++) {
            document.getElementById(`anova-g${i}`).textContent = `G${i}: ${window.App.statData.anova.groups[i-1].length}`;
          }

          const groups = window.App.statData.anova.groups.filter(g => g.length > 0);
          const groupMeans = groups.map(g => g.reduce((s,v)=>s+v,0)/g.length);
          this.drawStatisticsPlot('plot-anova', { groups, groupMeans }, 'anova');
        }
      }
    }
  }

  onStatisticsRightClick(type) {
    if (type === 'correlation') {
      if (window.App.statData.correlation.length > 0) {
        window.App.statData.correlation.pop();
        document.getElementById('corr-point-count').textContent = window.App.statData.correlation.length;
        const group1 = window.App.statData.correlation.map(p => p.x);
        const group2 = window.App.statData.correlation.map(p => p.y);
        this.drawStatisticsPlot('plot-correlation', { group1, group2 }, 'correlation');
      }
    } else if (type === 'ttest') {
      // Remove from the group that has the most recent addition
      if (window.App.statData.ttest.group2.length > 0) {
        window.App.statData.ttest.group2.pop();
        document.getElementById('ttest-n2').textContent = window.App.statData.ttest.group2.length;
      } else if (window.App.statData.ttest.group1.length > 0) {
        window.App.statData.ttest.group1.pop();
        document.getElementById('ttest-n1').textContent = window.App.statData.ttest.group1.length;
      }

      const mean1 = window.App.statData.ttest.group1.length > 0 ?
        window.App.statData.ttest.group1.reduce((s,v) => s+v, 0) / window.App.statData.ttest.group1.length : 0;
      const mean2 = window.App.statData.ttest.group2.length > 0 ?
        window.App.statData.ttest.group2.reduce((s,v) => s+v, 0) / window.App.statData.ttest.group2.length : 0;
      this.drawStatisticsPlot('plot-ttest', {...window.App.statData.ttest, mean1, mean2 }, 'ttest');
    } else if (type === 'anova') {
      const activeGroup = window.App.statData.anova.activeGroup;
      if (window.App.statData.anova.groups[activeGroup].length > 0) {
        window.App.statData.anova.groups[activeGroup].pop();
        for (let i = 1; i <= 5; i++) {
          document.getElementById(`anova-g${i}`).textContent = `G${i}: ${window.App.statData.anova.groups[i-1].length}`;
        }

        const groups = window.App.statData.anova.groups.filter(g => g.length > 0);
        const groupMeans = groups.map(g => g.reduce((s,v)=>s+v,0)/g.length);
        this.drawStatisticsPlot('plot-anova', { groups, groupMeans }, 'anova');
      }
    }
  }

  onMouseDown(e, canvas) {
    dlog('Mouse down event triggered!', canvas.id, e.button);
    if (!window.App) {
      derror('window.App is not defined!');
      return;
    }

    const [px, py] = this.getMousePos(e, canvas);
    const idx = this.findNearestPoint(px, py, canvas);
    dlog('Click position:', px, py, 'Nearest point index:', idx);

    if (e.button === 2 && idx >= 0) {
      window.App.removePoint(idx);
      return;
    }
    if (idx >= 0) {
      this.dragging = idx;
    } else {
      const [x, y] = AppUtils.pxToData(px, py, canvas);
      dlog('Adding new point at:', x, y);
      // For classification (logistic regression or nonlinear models), assign class based on selected group
      const isLogistic = window.App && window.App.activeRegTab === 'logistic';
      const isNLClassification = window.App && window.App.activeTab === 'nonlinear' &&
        window.App.nlTaskType[window.App.activeNLTab] === 'classification';
      const isClassification = isLogistic || isNLClassification;

      const label = isClassification ? (isLogistic ? window.App.getActiveLogisticGroup() : window.App.activeNLClass) : undefined;
      const userTest = !!(e.shiftKey && window.App.split && window.App.split.enabled);
      const point = isClassification ? {x, y, label} : {x, y};
      if (userTest) point.userTest = true;
      window.App.addPoint(point);
    }
  }

  onMouseMove(e, canvas) {
    if (this.dragging < 0) return;
    const [px, py] = this.getMousePos(e, canvas);
    const [x, y] = AppUtils.pxToData(px, py, canvas);
    const isLogistic = window.App && window.App.activeRegTab === 'logistic';
    const isNLClassification = window.App && window.App.activeTab === 'nonlinear' &&
      window.App.nlTaskType[window.App.activeNLTab] === 'classification';
    const isClassification = isLogistic || isNLClassification;
    const currentPoint = window.App.getPoints()[this.dragging];
    if (isClassification && currentPoint) {
      window.App.updatePoint(this.dragging, {x, y, label: currentPoint.label});
    } else {
      window.App.updatePoint(this.dragging, {x, y});
    }
  }

  getMousePos(e, canvas) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return [(e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY];
  }

  findNearestPoint(px, py, canvas) {
    let best = -1, bestDist = 400;
    const points = window.App.getPoints();
    for (let i = 0; i < points.length; i++) {
      const [x, y] = AppUtils.dataToPx(points[i].x, points[i].y, canvas);
      const d = (x - px) ** 2 + (y - py) ** 2;
      if (d < bestDist) { best = i; bestDist = d; }
    }
    return best;
  }
}

window.CanvasHandler = CanvasHandler;
