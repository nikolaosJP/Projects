// ============================================================================
// OLS SOLVER
// ============================================================================

class OLSSolver {
  solve(points) {
    const n = points.length;
    if (n < 2) throw new Error('Need at least 2 points!');

    let Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
    for (const p of points) {
      Sx += p.x; Sy += p.y; Sxx += p.x * p.x; Sxy += p.x * p.y;
    }

    const det = Sxx * n - Sx * Sx;
    if (Math.abs(det) < 1e-12) throw new Error('Singular matrix.');

    const m = (n * Sxy - Sx * Sy) / det;
    const b = (Sxx * Sy - Sx * Sxy) / det;

    return { m, b, fitted: true };
  }

  displaySteps(points, model) {
    const n = points.length;
    let Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
    for (const p of points) {
      Sx += p.x; Sy += p.y; Sxx += p.x * p.x; Sxy += p.x * p.y;
    }

    const steps = [
      `X^{\\top}X=\\begin{bmatrix}${Sxx.toFixed(3)}&${Sx.toFixed(3)}\\\\${Sx.toFixed(3)}&${n}\\end{bmatrix}`,
      `X^{\\top}y=\\begin{bmatrix}${Sxy.toFixed(3)}\\\\${Sy.toFixed(3)}\\end{bmatrix}`,
      `\\beta=(X^{\\top}X)^{-1}X^{\\top}y=\\begin{bmatrix}${model.m.toFixed(6)}\\\\${model.b.toFixed(6)}\\end{bmatrix}`
    ];

    const cont = document.getElementById('ols-steps');
    cont.innerHTML = '';
    steps.forEach(tex => {
      const div = document.createElement('div');
      div.className = 'calc-step';
      cont.appendChild(div);
      AppUtils.kRender(div, tex, true);
    });
    document.getElementById('ols-calculations').style.display = 'block';
  }

  updateDisplay(points, model, testPoints) {
    document.getElementById('ols-m').textContent = model.fitted ? model.m.toFixed(6) : '—';
    document.getElementById('ols-b').textContent = model.fitted ? model.b.toFixed(6) : '—';

    if (model.fitted && points.length) {
      const mse = AppUtils.calculateMSE(points, model.m, model.b);
      const r2 = AppUtils.calculateR2(points, model.m, model.b);
      const fmtMse = mse.toFixed(6);
      const fmtR2 = r2.toFixed(6);
      if (testPoints && testPoints.length) {
        const mseTest = AppUtils.calculateMSE(testPoints, model.m, model.b);
        const r2Test = AppUtils.calculateR2(testPoints, model.m, model.b);
        document.getElementById('ols-mse').innerHTML =
          `<span title="Train">${fmtMse}</span> <span style="color:var(--warning)" title="Test (${testPoints.length})">${mseTest.toFixed(6)}</span>`;
        document.getElementById('ols-r2').innerHTML =
          `<span title="Train">${fmtR2}</span> <span style="color:var(--warning)" title="Test (${testPoints.length})">${r2Test.toFixed(6)}</span>`;
      } else {
        document.getElementById('ols-mse').textContent = fmtMse;
        document.getElementById('ols-r2').textContent = fmtR2;
      }
    }
  }
}

// ============================================================================
// GRADIENT DESCENT
// ============================================================================

class GradientDescent {
  constructor() {
    this.reset();
  }

  reset() {
    this.initialized = false;
    this.currentStep = -1;
    this.iteration = 0;
    this.lossHistory = [];
    this.autoRunning = false;
    this.stepData = {};
    this.autoTimeoutId = null;
  }

  initialize(points) {
    if (points.length < 2) throw new Error('Add at least 2 points first!');
    
    const meanY = points.reduce((s, p) => s + p.y, 0) / points.length;
    const model = { m: 0, b: meanY, fitted: true };
    
    this.initialized = true;
    this.currentStep = 0;
    this.iteration = 0;
    this.lossHistory = [];
    this.stopAutoRun();
    this.stepData = {};

    document.getElementById('next-step').disabled = false;
    this.showAllSteps();
    this.showStep(0);
    AppUtils.kRender(document.getElementById('gd-init-formula'), 
                    `m_0=${model.m.toFixed(6)},\\; b_0=${model.b.toFixed(6)}`);
    this.updateStepDisplay();
    
    return model;
  }

  executeStep(points, model) {
    if (!this.initialized) return model;

    switch(this.currentStep) {
      case 0: return this.initStep(model);
      case 1: return this.predictionStep(points, model);
      case 2: return this.costStep(points, model);
      case 3: return this.gradientStep(points, model);
      case 4: return this.updateStep(model);
    }
    return model;
  }

  initStep(model) {
    this.showStep(0);
    AppUtils.kRender(document.getElementById('gd-init-formula'), 
                    `m_0=${model.m.toFixed(6)},\\; b_0=${model.b.toFixed(6)}`);
    this.currentStep++;
    this.updateStepDisplay();
    return model;
  }

  predictionStep(points, model) {
    this.stepData.predictions = [];
    this.stepData.errors = [];
    for (const p of points) {
      const pred = model.m * p.x + model.b;
      this.stepData.predictions.push(pred);
      this.stepData.errors.push(pred - p.y);
    }
    
    this.showStep(1);
    AppUtils.kRender(document.getElementById('gd-h-formula'), `h_i = m\\,x_i + b`);
    this.displayPredictionsTable(points);
    this.currentStep++;
    this.updateStepDisplay();
    return model;
  }

  costStep(points, model) {
    const sumSqErrors = this.stepData.errors.reduce((s, e) => s + e * e, 0);
    this.stepData.cost = sumSqErrors / (2 * points.length);
    this.lossHistory.push(this.stepData.cost);
    
    this.showStep(2);
    AppUtils.kRender(document.getElementById('gd-j-formula'), 
                    `J=\\tfrac{1}{2n}\\sum (h_i-y_i)^2 = ${this.stepData.cost.toFixed(6)}`);
    document.getElementById('cost-value').textContent = this.stepData.cost.toFixed(6);
    this.currentStep++;
    this.updateStepDisplay();
    return model;
  }

  gradientStep(points, model) {
    const n = points.length;
    this.stepData.gradM = 0;
    this.stepData.gradB = 0;
    for (let i = 0; i < n; i++) {
      this.stepData.gradM += points[i].x * this.stepData.errors[i] / n;
      this.stepData.gradB += this.stepData.errors[i] / n;
    }
    
    this.showStep(3);
    AppUtils.kRender(document.getElementById('gd-grads-formula'),
                    `\\frac{\\partial J}{\\partial m}=\\tfrac{1}{n}\\sum x_i(h_i-y_i),\\; \\frac{\\partial J}{\\partial b}=\\tfrac{1}{n}\\sum (h_i-y_i)`);
    document.getElementById('grad-m-val').textContent = this.stepData.gradM.toFixed(6);
    document.getElementById('grad-b-val').textContent = this.stepData.gradB.toFixed(6);
    this.currentStep++;
    this.updateStepDisplay();
    return model;
  }

  updateStep(model) {
    const lr = parseFloat(document.getElementById('lr').value) || 0.1;
    const newModel = {
      m: model.m - lr * this.stepData.gradM,
      b: model.b - lr * this.stepData.gradB,
      fitted: true
    };
    
    this.showStep(4);
    AppUtils.kRender(document.getElementById('gd-update-formula'),
                    `m\\leftarrow m-\\eta\\,\\tfrac{\\partial J}{\\partial m},\\; b\\leftarrow b-\\eta\\,\\tfrac{\\partial J}{\\partial b}`);
    document.getElementById('new-m-val').textContent = newModel.m.toFixed(6);
    document.getElementById('new-b-val').textContent = newModel.b.toFixed(6);
    
    this.iteration++;
    this.currentStep = 0;
    this.updateStepDisplay();
    return newModel;
  }

  displayPredictionsTable(points) {
    const tbody = document.getElementById('pred-body');
    tbody.innerHTML = '';
    for (let i = 0; i < points.length; i++) {
      const row = tbody.insertRow();
      row.insertCell(0).textContent = i + 1;
      row.insertCell(1).textContent = points[i].x.toFixed(3);
      row.insertCell(2).textContent = points[i].y.toFixed(3);
      row.insertCell(3).textContent = this.stepData.predictions[i].toFixed(3);
      row.insertCell(4).textContent = this.stepData.errors[i].toFixed(3);
    }
  }

  showAllSteps() {
    for (let i = 0; i <= 4; i++) {
      const el = document.getElementById(`step-${i}`);
      if (el) el.style.display = 'block';
    }
  }

  showStep(stepNum) {
    for (let i = 0; i <= 4; i++) {
      const el = document.getElementById(`step-${i}`);
      if (el) el.classList.remove('current-step');
    }
    const cur = document.getElementById(`step-${stepNum}`);
    if (cur) cur.classList.add('current-step');
  }

  updateStepDisplay() {
    const names = ['Initialization', 'Predictions', 'Cost', 'Gradients', 'Update'];
    document.getElementById('iteration').textContent = this.iteration;
    document.getElementById('current-step-name').textContent = 
      this.currentStep >= 0 ? names[this.currentStep] : 'Not Started';
  }

  runOneIteration(points, model) {
    const startIter = this.iteration;
    let newModel = model;
    let guard = 0;
    while (this.iteration === startIter && guard < 10) {
      newModel = this.executeStep(points, newModel);
      guard++;
    }
    return newModel;
  }

  startAutoRun(points, getCurrentModel, onUpdate) {
    this.autoRunning = !this.autoRunning;
    document.getElementById('auto-run').textContent = this.autoRunning ? 'Stop' : 'Auto';
    
    if (this.autoTimeoutId) {
      clearTimeout(this.autoTimeoutId);
      this.autoTimeoutId = null;
    }
    
    if (this.autoRunning) {
      const autoStep = () => {
        if (!this.autoRunning) return;
        
        const currentModel = getCurrentModel();
        const newModel = this.runOneIteration(points, currentModel);
        onUpdate(newModel);
        
        this.autoTimeoutId = setTimeout(autoStep, 50);
      };
      autoStep();
    }
  }

  stopAutoRun() {
    this.autoRunning = false;
    if (this.autoTimeoutId) {
      clearTimeout(this.autoTimeoutId);
      this.autoTimeoutId = null;
    }
    document.getElementById('auto-run').textContent = 'Auto';
  }
}

// ============================================================================
// MANUAL CALCULATOR
// ============================================================================

class ManualCalculator {
  calculate(points) {
    const n = points.length;
    if (n < 2) throw new Error('Need at least 2 points!');
    
    const meanX = points.reduce((s, p) => s + p.x, 0) / n;
    const meanY = points.reduce((s, p) => s + p.y, 0) / n;

    let ssxx = 0, ssyy = 0, ssxy = 0;
    for (const p of points) {
      const dx = p.x - meanX, dy = p.y - meanY;
      ssxx += dx * dx; ssyy += dy * dy; ssxy += dx * dy;
    }
    
    const b1 = ssxy / ssxx;
    const b0 = meanY - b1 * meanX;
    const model = {m: b1, b: b0, fitted: true};

    this.displayCalculations(points, meanX, meanY, ssxx, ssyy, ssxy, b1, b0);
    this.populateDataTable(points, meanX, meanY);
    
    return model;
  }

  displayCalculations(points, meanX, meanY, ssxx, ssyy, ssxy, b1, b0) {
    const n = points.length;
    
    // Calculate error metrics
    let ess = 0, rss = 0;
    for (const p of points) {
      const pred = b1 * p.x + b0;
      ess += (pred - meanY) ** 2;
      rss += (p.y - pred) ** 2;
    }
    
    const tss = ssyy;
    const r2 = ess / tss;
    const mse = rss / (n - 2);
    const rmse = Math.sqrt(mse);

    // Render formulas
    AppUtils.kRender(document.getElementById('mean-x-tex'), `\\bar{x}=\\tfrac{\\sum x_i}{n}=${meanX.toFixed(6)}`);
    AppUtils.kRender(document.getElementById('mean-y-tex'), `\\bar{y}=\\tfrac{\\sum y_i}{n}=${meanY.toFixed(6)}`);
    AppUtils.kRender(document.getElementById('ssxx-tex'), `SS_{xx}=\\sum(x_i-\\bar{x})^2=${ssxx.toFixed(6)}`);
    AppUtils.kRender(document.getElementById('ssyy-tex'), `SS_{yy}=\\sum(y_i-\\bar{y})^2=${ssyy.toFixed(6)}`);
    AppUtils.kRender(document.getElementById('ssxy-tex'), `SS_{xy}=\\sum(x_i-\\bar{x})(y_i-\\bar{y})=${ssxy.toFixed(6)}`);
    AppUtils.kRender(document.getElementById('b1-tex'), `b_1=\\tfrac{SS_{xy}}{SS_{xx}}=${b1.toFixed(6)}`);
    AppUtils.kRender(document.getElementById('b0-tex'), `b_0=\\bar{y}-b_1\\bar{x}=${b0.toFixed(6)}`);
    AppUtils.kRender(document.getElementById('tss-tex'), `TSS=\\sum(y_i-\\bar{y})^2=${tss.toFixed(6)}`);
    AppUtils.kRender(document.getElementById('ess-tex'), `ESS=\\sum(\\hat y_i-\\bar{y})^2=${ess.toFixed(6)}`);
    AppUtils.kRender(document.getElementById('rss-tex'), `RSS=\\sum(y_i-\\hat y_i)^2=${rss.toFixed(6)}`);
    AppUtils.kRender(document.getElementById('r2-tex'), `R^2=\\tfrac{ESS}{TSS}=${r2.toFixed(6)}`);
    AppUtils.kRender(document.getElementById('mse-tex'), `MSE=\\tfrac{RSS}{n-2}=${mse.toFixed(6)}`);
    AppUtils.kRender(document.getElementById('rmse-tex'), `RMSE=\\sqrt{MSE}=${rmse.toFixed(6)}`);

    document.getElementById('final-b1').textContent = b1.toFixed(6);
    document.getElementById('final-b0').textContent = b0.toFixed(6);
    document.getElementById('manual-calculations').style.display = 'block';
  }

  populateDataTable(points, meanX, meanY) {
    const tbody = document.getElementById('manual-body');
    tbody.innerHTML = '';
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const dx = p.x - meanX, dy = p.y - meanY;
      const row = tbody.insertRow();
      row.insertCell(0).textContent = i + 1;
      row.insertCell(1).textContent = p.x.toFixed(3);
      row.insertCell(2).textContent = p.y.toFixed(3);
      row.insertCell(3).textContent = dx.toFixed(3);
      row.insertCell(4).textContent = dy.toFixed(3);
      row.insertCell(5).textContent = (dx * dx).toFixed(3);
      row.insertCell(6).textContent = (dy * dy).toFixed(3);
      row.insertCell(7).textContent = (dx * dy).toFixed(3);
    }
  }
}

// ============================================================================
// LOGISTIC REGRESSION
// ============================================================================

class LogisticRegression {
  constructor() {
    this.reset();
  }

  reset() {
    this.initialized = false;
    this.currentStep = -1;
    this.iteration = 0;
    this.lossHistory = [];
    this.autoRunning = false;
    this.stepData = {};
    this.autoTimeoutId = null;
  }

  sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  initialize(points) {
    if (points.length < 2) throw new Error('Add at least 2 points first!');

    const model = { m: 0, b: 0, fitted: true };

    this.initialized = true;
    this.currentStep = 0;
    this.iteration = 0;
    this.lossHistory = [];
    this.stopAutoRun();
    this.stepData = {};

    document.getElementById('next-step-lr').disabled = false;
    this.showAllSteps();
    this.showStep(0);
    AppUtils.kRender(document.getElementById('lr-init-formula'),
                    `m_0=${model.m.toFixed(6)},\\; b_0=${model.b.toFixed(6)}`);
    this.updateStepDisplay();

    return model;
  }

  executeStep(points, model) {
    if (!this.initialized) return model;

    switch(this.currentStep) {
      case 0: return this.initStep(model);
      case 1: return this.predictionStep(points, model);
      case 2: return this.costStep(points, model);
      case 3: return this.gradientStep(points, model);
      case 4: return this.updateStep(model);
    }
    return model;
  }

  initStep(model) {
    this.showStep(0);
    AppUtils.kRender(document.getElementById('lr-init-formula'),
                    `m_0=${model.m.toFixed(6)},\\; b_0=${model.b.toFixed(6)}`);
    this.currentStep++;
    this.updateStepDisplay();
    return model;
  }

  predictionStep(points, model) {
    this.stepData.predictions = [];
    this.stepData.errors = [];
    for (const p of points) {
      const z = model.m * p.x + model.b;
      const pred = this.sigmoid(z);
      this.stepData.predictions.push(pred);
      this.stepData.errors.push(pred - p.label);
    }

    this.showStep(1);
    AppUtils.kRender(document.getElementById('lr-h-formula'),
                    `h_i = \\sigma(m\\,x_i + b) = \\frac{1}{1+e^{-(mx_i+b)}}`);
    this.displayPredictionsTable(points);
    this.currentStep++;
    this.updateStepDisplay();
    return model;
  }

  costStep(points, model) {
    let cost = 0;
    const n = points.length;
    for (let i = 0; i < n; i++) {
      const h = this.stepData.predictions[i];
      const y = points[i].label;
      cost += -(y * Math.log(h + 1e-10) + (1 - y) * Math.log(1 - h + 1e-10));
    }
    this.stepData.cost = cost / n;
    this.lossHistory.push(this.stepData.cost);

    this.showStep(2);
    AppUtils.kRender(document.getElementById('lr-j-formula'),
                    `J=-\\tfrac{1}{n}\\sum [y_i\\log(h_i)+(1-y_i)\\log(1-h_i)] = ${this.stepData.cost.toFixed(6)}`);
    document.getElementById('lr-cost-value').textContent = this.stepData.cost.toFixed(6);
    this.currentStep++;
    this.updateStepDisplay();
    return model;
  }

  gradientStep(points, model) {
    const n = points.length;
    this.stepData.gradM = 0;
    this.stepData.gradB = 0;
    for (let i = 0; i < n; i++) {
      this.stepData.gradM += points[i].x * this.stepData.errors[i] / n;
      this.stepData.gradB += this.stepData.errors[i] / n;
    }

    this.showStep(3);
    AppUtils.kRender(document.getElementById('lr-grads-formula'),
                    `\\frac{\\partial J}{\\partial m}=\\tfrac{1}{n}\\sum x_i(h_i-y_i),\\; \\frac{\\partial J}{\\partial b}=\\tfrac{1}{n}\\sum (h_i-y_i)`);
    document.getElementById('lr-grad-m-val').textContent = this.stepData.gradM.toFixed(6);
    document.getElementById('lr-grad-b-val').textContent = this.stepData.gradB.toFixed(6);
    this.currentStep++;
    this.updateStepDisplay();
    return model;
  }

  updateStep(model) {
    const lr = parseFloat(document.getElementById('lr-lr').value) || 0.1;
    const newModel = {
      m: model.m - lr * this.stepData.gradM,
      b: model.b - lr * this.stepData.gradB,
      fitted: true
    };

    this.showStep(4);
    AppUtils.kRender(document.getElementById('lr-update-formula'),
                    `m\\leftarrow m-\\eta\\,\\tfrac{\\partial J}{\\partial m},\\; b\\leftarrow b-\\eta\\,\\tfrac{\\partial J}{\\partial b}`);
    document.getElementById('lr-new-m-val').textContent = newModel.m.toFixed(6);
    document.getElementById('lr-new-b-val').textContent = newModel.b.toFixed(6);

    this.iteration++;
    this.currentStep = 0;
    this.updateStepDisplay();
    return newModel;
  }

  displayPredictionsTable(points) {
    const tbody = document.getElementById('lr-pred-body');
    tbody.innerHTML = '';
    for (let i = 0; i < points.length; i++) {
      const row = tbody.insertRow();
      row.insertCell(0).textContent = i + 1;
      row.insertCell(1).textContent = points[i].x.toFixed(3);
      row.insertCell(2).textContent = points[i].label;
      row.insertCell(3).textContent = this.stepData.predictions[i].toFixed(3);
      row.insertCell(4).textContent = this.stepData.errors[i].toFixed(3);
    }
  }

  showAllSteps() {
    for (let i = 0; i <= 4; i++) {
      const el = document.getElementById(`lr-step-${i}`);
      if (el) el.style.display = 'block';
    }
  }

  showStep(stepNum) {
    for (let i = 0; i <= 4; i++) {
      const el = document.getElementById(`lr-step-${i}`);
      if (el) el.classList.remove('current-step');
    }
    const cur = document.getElementById(`lr-step-${stepNum}`);
    if (cur) cur.classList.add('current-step');
  }

  updateStepDisplay() {
    const names = ['Initialization', 'Predictions', 'Cost', 'Gradients', 'Update'];
    document.getElementById('lr-iteration').textContent = this.iteration;
    document.getElementById('lr-current-step-name').textContent =
      this.currentStep >= 0 ? names[this.currentStep] : 'Not Started';
  }

  runOneIteration(points, model) {
    const startIter = this.iteration;
    let newModel = model;
    let guard = 0;
    while (this.iteration === startIter && guard < 10) {
      newModel = this.executeStep(points, newModel);
      guard++;
    }
    return newModel;
  }

  startAutoRun(points, getCurrentModel, onUpdate) {
    this.autoRunning = !this.autoRunning;
    document.getElementById('lr-auto-run').textContent = this.autoRunning ? 'Stop' : 'Auto';

    if (this.autoTimeoutId) {
      clearTimeout(this.autoTimeoutId);
      this.autoTimeoutId = null;
    }

    if (this.autoRunning) {
      const autoStep = () => {
        if (!this.autoRunning) return;

        const currentModel = getCurrentModel();
        const newModel = this.runOneIteration(points, currentModel);
        onUpdate(newModel);

        this.autoTimeoutId = setTimeout(autoStep, 50);
      };
      autoStep();
    }
  }

  stopAutoRun() {
    this.autoRunning = false;
    if (this.autoTimeoutId) {
      clearTimeout(this.autoTimeoutId);
      this.autoTimeoutId = null;
    }
    const btn = document.getElementById('lr-auto-run');
    if (btn) btn.textContent = 'Auto';
  }

  calculateMetrics(points, model, threshold = 0.5) {
    if (!points.length) return null;

    let tp = 0, fp = 0, tn = 0, fn = 0;
    const predictions = [];

    for (const p of points) {
      const z = model.m * p.x + model.b;
      const prob = this.sigmoid(z);
      const pred = prob >= threshold ? 1 : 0;
      predictions.push({ actual: p.label, predicted: pred, probability: prob });

      if (p.label === 1 && pred === 1) tp++;
      else if (p.label === 0 && pred === 1) fp++;
      else if (p.label === 0 && pred === 0) tn++;
      else if (p.label === 1 && pred === 0) fn++;
    }

    const accuracy = (tp + tn) / points.length;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0; // Also known as sensitivity
    const specificity = tn + fp > 0 ? tn / (tn + fp) : 0;
    const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

    return {
      tp, fp, tn, fn,
      accuracy, precision, recall, specificity, f1,
      predictions
    };
  }

  calculateYoudenThreshold(points, model) {
    if (!points.length) return 0.5;

    // Calculate Youden index for various thresholds
    let bestThreshold = 0.5;
    let bestYouden = -1;

    for (let t = 0; t <= 1; t += 0.01) {
      const metrics = this.calculateMetrics(points, model, t);
      const youden = metrics.recall + metrics.specificity - 1;

      if (youden > bestYouden) {
        bestYouden = youden;
        bestThreshold = t;
      }
    }

    return bestThreshold;
  }
}

// Export classes to global scope
window.OLSSolver = OLSSolver;
window.GradientDescent = GradientDescent;
window.ManualCalculator = ManualCalculator;
window.LogisticRegression = LogisticRegression;
