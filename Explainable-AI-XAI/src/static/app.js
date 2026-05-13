// ============================================================================
// MAIN APPLICATION
// ============================================================================

class App {
  constructor() {
    this.points = [];
    this.activeTab = 'linear';
    this.activeRegTab = 'ols';
    this.activeNLTab = 'dt';
    this.activeStatTab = 'correlation';
    this.showResiduals = true;
    this.activeLogisticGroup = 0; // Currently selected group for logistic regression (0 or 1)
    this.activeNLClass = 0; // Currently selected class for nonlinear classification (0 or 1)
    this.nlTaskType = { dt: 'regression', rf: 'regression', xgb: 'regression', nn: 'regression', rnn: 'regression', lstm: 'regression' }; // Task type for each NL model

    this.models = {
      ols: {m: 0, b: 0, fitted: false},
      gd: {m: 0, b: 0, fitted: false},
      manual: {m: 0, b: 0, fitted: false},
      logistic: {m: 0, b: 0, fitted: false},
      dt: {fitted: false, tree: null},
      rf: {fitted: false, trees: [], oobIndices: []},
      xgb: {fitted: false, trees: [], basePrediction: 0, iteration: 0},
      nn: {fitted: false, weights: [], biases: [], epoch: 0},
      rnn: {fitted: false, predictions: []},
      lstm: {fitted: false, predictions: []}
    };

    // Statistics data storage
    this.statData = {
      correlation: [],
      ttest: { group1: [], group2: [], activeGroup: 0, testType: 'two-tailed' },
      anova: { groups: [[], [], [], [], []], activeGroup: 0 }
    };

    this.canvas = new CanvasHandler();
    this.olsSolver = new OLSSolver();
    this.gradientDescent = new GradientDescent();
    this.manualCalculator = new ManualCalculator();
    this.logisticRegression = new LogisticRegression();
    this.statisticsCalculator = new StatisticsCalculator();
    this.decisionTree = new DecisionTree();
    this.randomForest = new RandomForest();
    this.xgboost = new XGBoost();
    this.neuralNetwork = new NeuralNetwork();
    this.recommendationSystem = new RecommendationSystem();
    this.contentBasedFiltering = new ContentBasedFiltering(this.recommendationSystem);
    this.collaborativeFiltering = new CollaborativeFiltering(this.recommendationSystem);
    this.clusteringManager = new ClusteringManager();
    this.kMeansClustering = new KMeansClustering(this.clusteringManager);
    this.hierarchicalClustering = new HierarchicalClustering(this.clusteringManager);
    this.clusteringPoints = [];

    this.transformerDemo = new TransformerDemo();
    this.ragDemo = new RAGDemo();
    this.activeLLMTab = 'transformer';

    this.rnn = new VanillaRNN();
    this.lstm = new LSTM();

    this.setupEventHandlers();
    this.initializeFormulas();
    this.redrawCanvas();
    AppUtils.kFlush();
  }

  setupEventHandlers() {
    // Main tabs
    document.querySelectorAll('.tab[data-tab]').forEach(tab => {
      tab.addEventListener('click', () => {
        if (tab.dataset.tab) this.switchTab(tab.dataset.tab);
      });
    });

    // Regression subtabs
    document.querySelectorAll('.tab[data-regtab]').forEach(tab => {
      tab.addEventListener('click', () => {
        if (tab.dataset.regtab) this.switchRegTab(tab.dataset.regtab);
      });
    });

    // Nonlinear Models subtabs
    document.querySelectorAll('.tab[data-nltab]').forEach(tab => {
      tab.addEventListener('click', () => {
        if (tab.dataset.nltab) this.switchNLTab(tab.dataset.nltab);
      });
    });

    // Statistics subtabs
    document.querySelectorAll('.tab[data-subtab]').forEach(tab => {
      tab.addEventListener('click', () => {
        if (tab.dataset.subtab) this.switchStatTab(tab.dataset.subtab);
      });
    });

    // LLM subtabs
    document.querySelectorAll('.tab[data-llmtab]').forEach(tab => {
      tab.addEventListener('click', () => {
        if (tab.dataset.llmtab) this.switchLLMTab(tab.dataset.llmtab);
      });
    });

    // Transformer controls
    const trInit = document.getElementById('tr-init');
    if (trInit) trInit.onclick = () => this.initTransformer();
    const trNext = document.getElementById('tr-next');
    if (trNext) trNext.onclick = () => this.nextTransformerStep();
    const trRun = document.getElementById('tr-run');
    if (trRun) trRun.onclick = () => this.runTransformer();
    const trReset = document.getElementById('tr-reset');
    if (trReset) trReset.onclick = () => this.resetTransformer();

    // RAG controls
    const ragRun = document.getElementById('rag-run');
    if (ragRun) ragRun.onclick = () => this.runRAG();
    const ragReset = document.getElementById('rag-reset');
    if (ragReset) ragReset.onclick = () => this.resetRAG();

    // OLS Controls
    document.getElementById('solve-ols').onclick = () => this.solveOLS();

    // GD Controls
    document.getElementById('init-gd').onclick = () => this.initializeGD();
    document.getElementById('next-step').onclick = () => this.nextGDStep();
    document.getElementById('auto-run').onclick = () => this.autoRunGD();
    document.getElementById('reset-gd').onclick = () => this.resetGD();

    // Manual Controls
    document.getElementById('calculate-manual').onclick = () => this.calculateManual();

    // Logistic Regression Controls
    document.getElementById('init-lr').onclick = () => this.initializeLR();
    document.getElementById('next-step-lr').onclick = () => this.nextLRStep();
    document.getElementById('lr-auto-run').onclick = () => this.autoRunLR();
    document.getElementById('reset-lr').onclick = () => this.resetLR();
    document.getElementById('lr-threshold').oninput = (e) => this.updateLRMetrics(parseFloat(e.target.value));
    document.getElementById('lr-youden-btn').onclick = () => this.useYoudenThreshold();
    document.getElementById('lr-group0-btn').onclick = () => this.setLogisticGroup(0);
    document.getElementById('lr-group1-btn').onclick = () => this.setLogisticGroup(1);

    // Decision Tree Controls
    document.getElementById('fit-dt').onclick = () => this.fitDT();
    document.getElementById('reset-dt').onclick = () => this.resetDT();
    document.getElementById('dt-task-reg').onclick = () => this.setNLTask('dt', 'regression');
    document.getElementById('dt-task-cls').onclick = () => this.setNLTask('dt', 'classification');
    document.getElementById('dt-class0-btn').onclick = () => this.setNLClass('dt', 0);
    document.getElementById('dt-class1-btn').onclick = () => this.setNLClass('dt', 1);

    // Random Forest Controls
    document.getElementById('fit-rf').onclick = () => this.fitRF();
    document.getElementById('reset-rf').onclick = () => this.resetRF();
    document.getElementById('rf-task-reg').onclick = () => this.setNLTask('rf', 'regression');
    document.getElementById('rf-task-cls').onclick = () => this.setNLTask('rf', 'classification');
    document.getElementById('rf-class0-btn').onclick = () => this.setNLClass('rf', 0);
    document.getElementById('rf-class1-btn').onclick = () => this.setNLClass('rf', 1);

    // XGBoost Controls
    document.getElementById('init-xgb').onclick = () => this.initXGB();
    document.getElementById('next-boost-xgb').onclick = () => this.nextBoostXGB();
    document.getElementById('auto-xgb').onclick = () => this.autoXGB();
    document.getElementById('reset-xgb').onclick = () => this.resetXGB();
    document.getElementById('xgb-task-reg').onclick = () => this.setNLTask('xgb', 'regression');
    document.getElementById('xgb-task-cls').onclick = () => this.setNLTask('xgb', 'classification');
    document.getElementById('xgb-class0-btn').onclick = () => this.setNLClass('xgb', 0);
    document.getElementById('xgb-class1-btn').onclick = () => this.setNLClass('xgb', 1);

    // Neural Network Controls
    document.getElementById('init-nn').onclick = () => this.initNN();
    document.getElementById('next-epoch-nn').onclick = () => this.nextEpochNN();
    document.getElementById('auto-nn').onclick = () => this.autoNN();
    document.getElementById('reset-nn').onclick = () => this.resetNN();
    document.getElementById('nn-task-reg').onclick = () => this.setNLTask('nn', 'regression');
    document.getElementById('nn-task-cls').onclick = () => this.setNLTask('nn', 'classification');
    document.getElementById('nn-class0-btn').onclick = () => this.setNLClass('nn', 0);
    document.getElementById('nn-class1-btn').onclick = () => this.setNLClass('nn', 1);

    // RNN Controls
    const initRnn = document.getElementById('init-rnn');
    if (initRnn) initRnn.onclick = () => this.initRNN();
    const nextRnn = document.getElementById('next-step-rnn');
    if (nextRnn) nextRnn.onclick = () => this.nextStepRNN();
    const runRnn = document.getElementById('run-rnn');
    if (runRnn) runRnn.onclick = () => this.runRNN();
    const trainRnn = document.getElementById('train-rnn');
    if (trainRnn) trainRnn.onclick = () => this.trainRNN();
    const resetRnn = document.getElementById('reset-rnn');
    if (resetRnn) resetRnn.onclick = () => this.resetRNN();

    // LSTM Controls
    const initLstm = document.getElementById('init-lstm');
    if (initLstm) initLstm.onclick = () => this.initLSTM();
    const nextLstm = document.getElementById('next-step-lstm');
    if (nextLstm) nextLstm.onclick = () => this.nextStepLSTM();
    const runLstm = document.getElementById('run-lstm');
    if (runLstm) runLstm.onclick = () => this.runLSTM();
    const trainLstm = document.getElementById('train-lstm');
    if (trainLstm) trainLstm.onclick = () => this.trainLSTM();
    const resetLstm = document.getElementById('reset-lstm');
    if (resetLstm) resetLstm.onclick = () => this.resetLSTM();

    // Statistics Controls - Correlation
    document.getElementById('calc-correlation').onclick = () => this.calculateCorrelation();
    document.getElementById('clear-correlation').onclick = () => this.clearStatData('correlation');
    document.getElementById('sample-correlation').onclick = () => this.sampleStatData('correlation');

    // Statistics Controls - t-Test
    document.getElementById('calc-ttest').onclick = () => this.calculateTTest();
    document.getElementById('clear-ttest').onclick = () => this.clearStatData('ttest');
    document.getElementById('sample-ttest').onclick = () => this.sampleStatData('ttest');
    document.getElementById('ttest-group1-btn').onclick = () => this.setTTestGroup(0);
    document.getElementById('ttest-group2-btn').onclick = () => this.setTTestGroup(1);

    // t-Test type selection
    document.querySelectorAll('input[name="ttest-type"]').forEach(radio => {
      radio.addEventListener('change', (e) => {
        this.statData.ttest.testType = e.target.value;
      });
    });

    // Statistics Controls - ANOVA
    document.getElementById('calc-anova').onclick = () => this.calculateANOVA();
    document.getElementById('clear-anova').onclick = () => this.clearStatData('anova');
    document.getElementById('sample-anova').onclick = () => this.sampleStatData('anova');
    for (let i = 1; i <= 5; i++) {
      document.getElementById(`anova-g${i}-btn`).onclick = () => this.setAnovaGroup(i - 1);
    }

    // Recommendation Systems Controls - Data Table
    document.getElementById('recsys-add-user').onclick = () => this.addRecSysUser();
    document.getElementById('recsys-add-movie').onclick = () => this.addRecSysMovie();
    document.getElementById('recsys-reset-data').onclick = () => this.resetRecSysData();
    document.getElementById('recsys-random-data').onclick = () => this.randomRecSysData();

    // Movie Features Table Controls
    document.getElementById('features-add-genre').onclick = () => this.addMovieFeature();
    document.getElementById('features-reset-data').onclick = () => this.resetMovieFeatures();
    document.getElementById('features-random-data').onclick = () => this.randomMovieFeatures();

    // Recommendation Systems Controls - Content-Based Filtering
    document.getElementById('compute-cb').onclick = () => this.computeContentBased();

    // Recommendation Systems Controls - Collaborative Filtering
    document.getElementById('init-cf').onclick = () => this.initCF();
    document.getElementById('next-epoch-cf').onclick = () => this.nextEpochCF();
    document.getElementById('auto-cf').onclick = () => this.autoCF();
    document.getElementById('reset-cf').onclick = () => this.resetCF();

    // Clustering Controls - Canvas
    document.getElementById('clustering-clear').onclick = () => this.clearClustering();
    document.getElementById('clustering-sample').onclick = () => this.sampleClustering();

    // Clustering canvas events
    const clusteringCanvas = document.getElementById('plot-clustering');
    if (clusteringCanvas) {
      clusteringCanvas.addEventListener('contextmenu', e => e.preventDefault());
      clusteringCanvas.addEventListener('mousedown', e => this.onClusteringCanvasClick(e, clusteringCanvas));
    }

    // Clustering Controls - K-Means
    document.getElementById('init-kmeans').onclick = () => this.initKMeans();
    document.getElementById('next-kmeans').onclick = () => this.nextKMeansIter();
    document.getElementById('auto-kmeans').onclick = () => this.autoKMeans();
    document.getElementById('reset-kmeans').onclick = () => this.resetKMeans();

    // Clustering Controls - Hierarchical
    document.getElementById('compute-hierarchical').onclick = () => this.computeHierarchical();
    document.getElementById('reset-hierarchical').onclick = () => this.resetHierarchical();

    // General Controls
    document.getElementById('clear').onclick = () => this.clear();
    document.getElementById('sample').onclick = () => this.generateSample();
    const residualsCheckbox = document.getElementById('residuals');
    if (residualsCheckbox) {
      residualsCheckbox.onchange = e => {
        this.showResiduals = e.target.checked;
        this.redrawCanvas();
      };
    }

    // Train/test split controls
    this.split = { enabled: false, fraction: 0.2, salt: 1 };
    const splitEnabled = document.getElementById('split-enabled');
    const splitFraction = document.getElementById('split-fraction');
    const splitLabel = document.getElementById('split-fraction-label');
    const splitReshuffle = document.getElementById('split-reshuffle');
    if (splitEnabled) {
      splitEnabled.onchange = e => {
        this.split.enabled = e.target.checked;
        const legend = document.getElementById('split-legend');
        if (legend) legend.style.display = this.split.enabled ? 'inline-flex' : 'none';
        this.recomputeSplit();
        this.redrawCanvas();
      };
    }
    if (splitFraction) {
      splitFraction.oninput = e => {
        this.split.fraction = parseFloat(e.target.value);
        if (splitLabel) splitLabel.textContent = `${Math.round(this.split.fraction * 100)}%`;
        if (this.split.enabled) {
          this.recomputeSplit();
          this.redrawCanvas();
        }
      };
    }
    if (splitReshuffle) {
      splitReshuffle.onclick = () => {
        this.split.salt = (this.split.salt + 1) | 0;
        if (this.split.enabled) {
          this.recomputeSplit();
          this.redrawCanvas();
        }
      };
    }

    // Theme toggle
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
      themeBtn.onclick = () => {
        const cur = document.documentElement.getAttribute('data-theme') || 'light';
        const next = cur === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        try { localStorage.setItem('xai-theme', next); } catch (e) {}
        this.redrawCanvas();
      };
    }

    // Keyboard controls
    document.addEventListener('keydown', (e) => {
      // ANOVA group selection
      if (this.activeTab === 'stats' && this.activeStatTab === 'anova') {
        const key = parseInt(e.key);
        if (key >= 1 && key <= 5) {
          this.statData.anova.activeGroup = key - 1;
          document.getElementById('anova-active-group').textContent = key;
          return;
        }
      }
      // Tab navigation with arrow keys when a tab has focus
      const target = e.target;
      if (target && target.classList && target.classList.contains('tab') && target.dataset.tab) {
        const tabs = Array.from(document.querySelectorAll('.tab[data-tab]'));
        const idx = tabs.indexOf(target);
        if (idx >= 0) {
          let next = idx;
          if (e.key === 'ArrowRight') next = (idx + 1) % tabs.length;
          else if (e.key === 'ArrowLeft') next = (idx - 1 + tabs.length) % tabs.length;
          else if (e.key === 'Enter' || e.key === ' ') { target.click(); e.preventDefault(); return; }
          if (next !== idx) {
            tabs[next].focus();
            tabs[next].click();
            e.preventDefault();
          }
        }
      }
    });
  }

  // Public: return a fresh redraw of the active tab
  redraw() {
    if (this.redrawCanvas) this.redrawCanvas();
  }

  // Deterministic 0-1 hash from (index, salt, pointCount)
  _splitHash(i, n) {
    let h = (i * 2654435761) ^ (this.split.salt * 374761393) ^ (n * 668265263);
    h = (h ^ (h >>> 13)) * 1274126177;
    h = (h ^ (h >>> 16)) >>> 0;
    return h / 4294967295;
  }

  // Mark each point with isTest = true/false based on current split state
  recomputeSplit() {
    if (!this.points) return;
    const n = this.points.length;
    if (!this.split.enabled || n < 4) {
      this.points.forEach(p => { p.isTest = false; });
      return;
    }
    this.points.forEach((p, i) => {
      // Use the user-flagged value if the point was added with shift; otherwise hash
      if (p.userTest) { p.isTest = true; return; }
      p.isTest = this._splitHash(i, n) < this.split.fraction;
    });
  }

  getTrainPoints() {
    if (!this.points) return [];
    if (!this.split.enabled) return this.points;
    return this.points.filter(p => !p.isTest);
  }

  getTestPoints() {
    if (!this.points) return [];
    if (!this.split.enabled) return [];
    return this.points.filter(p => p.isTest);
  }

  switchTab(tab) {
    document.querySelectorAll('.tab[data-tab]').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    const tabButton = document.querySelector(`[data-tab="${tab}"]`);
    if (tabButton) tabButton.classList.add('active');
    const tabContent = document.getElementById(`tab-${tab}`);
    if (tabContent) tabContent.classList.add('active');
    this.activeTab = tab;

    // Hide general controls for tabs that don't use the shared point canvas
    const generalControls = document.getElementById('general-controls-panel');
    if (generalControls) {
      const noCanvas = (tab === 'recsys' || tab === 'clustering' || tab === 'llms');
      generalControls.style.display = noCanvas ? 'none' : 'block';
    }

    // Redraw canvas when switching tabs
    this.redrawCanvas();
  }

  switchRegTab(regtab) {
    document.querySelectorAll('.tab[data-regtab]').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.regtab-content').forEach(t => t.classList.remove('active'));
    const tabButton = document.querySelector(`[data-regtab="${regtab}"]`);
    if (tabButton) tabButton.classList.add('active');
    const tabContent = document.getElementById(`regtab-${regtab}`);
    if (tabContent) tabContent.classList.add('active');
    this.activeRegTab = regtab;

    // Use redrawCanvas for consistency
    this.redrawCanvas();
  }

  switchStatTab(subtab) {
    document.querySelectorAll('.tab[data-subtab]').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.subtab-content').forEach(t => t.classList.remove('active'));
    const tabButton = document.querySelector(`[data-subtab="${subtab}"]`);
    if (tabButton) tabButton.classList.add('active');
    const tabContent = document.getElementById(`subtab-${subtab}`);
    if (tabContent) tabContent.classList.add('active');
    this.activeStatTab = subtab;
  }

  switchNLTab(nltab) {
    document.querySelectorAll('.tab[data-nltab]').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.nltab-content').forEach(t => t.classList.remove('active'));
    const tabButton = document.querySelector(`[data-nltab="${nltab}"]`);
    if (tabButton) tabButton.classList.add('active');
    const tabContent = document.getElementById(`nltab-${nltab}`);
    if (tabContent) tabContent.classList.add('active');
    this.activeNLTab = nltab;

    // Use redrawCanvas for consistency
    this.redrawCanvas();
  }

  switchLLMTab(llmtab) {
    document.querySelectorAll('.tab[data-llmtab]').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.llmtab-content').forEach(t => t.classList.remove('active'));
    const btn = document.querySelector(`[data-llmtab="${llmtab}"]`);
    if (btn) btn.classList.add('active');
    const content = document.getElementById(`llmtab-${llmtab}`);
    if (content) content.classList.add('active');
    this.activeLLMTab = llmtab;
  }

  // ---- Transformer ----
  initTransformer() {
    const sentence = document.getElementById('tr-sentence').value;
    let dim = parseInt(document.getElementById('tr-dim').value);
    let seed = parseInt(document.getElementById('tr-seed').value);
    if (!Number.isFinite(dim) || dim < 2) dim = 2;
    if (dim > 8) dim = 8;
    if (!Number.isFinite(seed)) seed = 42;
    this.transformerDemo.resetUI();
    const ok = this.transformerDemo.initialize(sentence, dim, seed);
    if (!ok) return;
    document.getElementById('tr-next').disabled = false;
    document.getElementById('tr-run').disabled = false;
    document.getElementById('tr-step-name').textContent = 'Initialized';
  }

  nextTransformerStep() {
    const name = this.transformerDemo.nextStep();
    if (!name) {
      window.showToast('Pipeline complete. Click Reset to start again.', 'info');
      document.getElementById('tr-next').disabled = true;
    }
  }

  runTransformer() {
    if (!this.transformerDemo.state) {
      this.initTransformer();
    }
    this.transformerDemo.runAll();
    document.getElementById('tr-next').disabled = true;
    document.getElementById('tr-step-name').textContent = 'Done';
  }

  resetTransformer() {
    this.transformerDemo.reset();
    this.transformerDemo.resetUI();
    document.getElementById('tr-next').disabled = true;
    document.getElementById('tr-run').disabled = true;
  }

  // ---- RAG ----
  runRAG() {
    const corpus = document.getElementById('rag-corpus').value;
    const query = document.getElementById('rag-query').value;
    let topK = parseInt(document.getElementById('rag-topk').value);
    if (!Number.isFinite(topK) || topK < 1) topK = 1;
    this.ragDemo.run(corpus, query, topK);
  }

  resetRAG() {
    this.ragDemo.resetUI();
  }

  setNLTask(model, task) {
    this.nlTaskType[model] = task;
    // Update UI
    document.getElementById(`${model}-task-reg`).classList.toggle('success', task === 'regression');
    document.getElementById(`${model}-task-cls`).classList.toggle('success', task === 'classification');
    // Show/hide class selector
    const classSelector = document.getElementById(`${model}-class-selector`);
    if (classSelector) {
      classSelector.style.display = task === 'classification' ? 'flex' : 'none';
    }
    // Clear points when switching task type
    this.clear();
    // Redraw canvas
    this.canvas.drawNL(model, this.points, this.models, task);
  }

  setNLClass(model, classLabel) {
    this.activeNLClass = classLabel;
    // Update button styles - remove warning from both, add to selected
    document.getElementById(`${model}-class0-btn`).classList.remove('warning');
    document.getElementById(`${model}-class1-btn`).classList.remove('warning');
    document.getElementById(`${model}-class${classLabel}-btn`).classList.add('warning');
  }

  // Decision Tree methods
  fitDT() {
    try {
      const maxDepth = parseInt(document.getElementById('dt-max-depth').value);
      const minSamples = parseInt(document.getElementById('dt-min-samples').value);
      const taskType = this.nlTaskType.dt;

      const train = this.getTrainPoints();
      this.models.dt = this.decisionTree.fit(train, taskType, maxDepth, minSamples);

      // Display build steps
      const stepsDisplay = document.getElementById('dt-step-display');
      if (stepsDisplay && this.models.dt.buildSteps) {
        stepsDisplay.innerHTML = this.models.dt.buildSteps.join('');
        document.getElementById('dt-steps').style.display = 'block';
        // Render inline formulas
        AppUtils.kRenderInlineFormulas(stepsDisplay);
      }

      this.decisionTree.updateDisplay(train, this.models.dt, taskType);
      this.canvas.drawNL('dt', this.points, this.models, taskType);

      document.getElementById('dt-metrics').style.display = 'block';
      document.getElementById('dt-tree-viz').style.display = 'block';
      if (taskType === 'regression') {
        document.getElementById('dt-reg-metrics').style.display = 'grid';
        document.getElementById('dt-cls-metrics').style.display = 'none';
      } else {
        document.getElementById('dt-reg-metrics').style.display = 'none';
        document.getElementById('dt-cls-metrics').style.display = 'grid';
      }
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  resetDT() {
    this.models.dt = { fitted: false, tree: null };
    this.decisionTree.reset();
    document.getElementById('dt-steps').style.display = 'none';
    document.getElementById('dt-metrics').style.display = 'none';
    document.getElementById('dt-tree-viz').style.display = 'none';
    this.canvas.drawNL('dt', this.points, this.models, this.nlTaskType.dt);
  }

  // Random Forest methods
  fitRF() {
    try {
      const nTrees = parseInt(document.getElementById('rf-n-trees').value);
      const maxDepth = parseInt(document.getElementById('rf-max-depth').value);
      const minSamples = parseInt(document.getElementById('rf-min-samples').value);
      const maxFeatures = parseInt(document.getElementById('rf-max-features').value);
      const taskType = this.nlTaskType.rf;

      const train = this.getTrainPoints();
      this.models.rf = this.randomForest.fit(train, taskType, nTrees, maxDepth, minSamples, maxFeatures);

      // Display build steps
      const stepsDisplay = document.getElementById('rf-step-display');
      if (stepsDisplay && this.models.rf.buildSteps) {
        stepsDisplay.innerHTML = this.models.rf.buildSteps.join('');
        document.getElementById('rf-steps').style.display = 'block';
        // Render inline formulas
        AppUtils.kRenderInlineFormulas(stepsDisplay);
      }

      this.randomForest.updateDisplay(train, this.models.rf, taskType);
      this.canvas.drawNL('rf', this.points, this.models, taskType);

      document.getElementById('rf-metrics').style.display = 'block';
      document.getElementById('rf-trees-viz').style.display = 'block';
      if (taskType === 'regression') {
        document.getElementById('rf-reg-metrics').style.display = 'grid';
        document.getElementById('rf-cls-metrics').style.display = 'none';
      } else {
        document.getElementById('rf-reg-metrics').style.display = 'none';
        document.getElementById('rf-cls-metrics').style.display = 'grid';
      }
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  resetRF() {
    this.models.rf = { fitted: false, trees: [], oobIndices: [] };
    this.randomForest.reset();
    document.getElementById('rf-steps').style.display = 'none';
    document.getElementById('rf-metrics').style.display = 'none';
    document.getElementById('rf-trees-viz').style.display = 'none';
    this.canvas.drawNL('rf', this.points, this.models, this.nlTaskType.rf);
  }

  // XGBoost methods
  initXGB() {
    try {
      const taskType = this.nlTaskType.xgb;
      this.models.xgb = this.xgboost.initialize(this.getTrainPoints(), taskType);
      document.getElementById('next-boost-xgb').disabled = false;
      document.getElementById('xgb-iteration').textContent = '0';
      document.getElementById('xgb-tree-count').textContent = '0';

      // Show initial explanation with formulas
      const stepsDisplay = document.getElementById('xgb-step-display');
      if (stepsDisplay && this.models.xgb.initialExplanation) {
        stepsDisplay.innerHTML = this.models.xgb.initialExplanation;
        document.getElementById('xgb-steps').style.display = 'block';
        // Render inline formulas
        AppUtils.kRenderInlineFormulas(stepsDisplay);
      }

      this.canvas.drawNL('xgb', this.points, this.models, taskType);
      this.drawXGBLossPlot();
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  nextBoostXGB() {
    try {
      const lr = parseFloat(document.getElementById('xgb-lr').value);
      const maxDepth = parseInt(document.getElementById('xgb-max-depth').value);
      const lambda = parseFloat(document.getElementById('xgb-lambda').value);
      const gamma = parseFloat(document.getElementById('xgb-gamma').value);
      const subsample = parseFloat(document.getElementById('xgb-subsample').value);
      const taskType = this.nlTaskType.xgb;

      // If this is the first boost, add a separator after the initial explanation
      const stepsDisplay = document.getElementById('xgb-step-display');
      if (this.models.xgb.iteration === 0 && stepsDisplay) {
        stepsDisplay.innerHTML += '<br><strong>Boosting Iterations:</strong><br>';
      }

      const train = this.getTrainPoints();
      this.models.xgb = this.xgboost.executeBoost(train, this.models.xgb, taskType, lr, maxDepth, lambda, gamma, subsample);

      // Append step information
      if (stepsDisplay && this.models.xgb.currentStepDisplay) {
        stepsDisplay.innerHTML += this.models.xgb.currentStepDisplay;
        // Render inline formulas
        AppUtils.kRenderInlineFormulas(stepsDisplay);
      }

      this.xgboost.updateDisplay(train, this.models.xgb, taskType);
      this.canvas.drawNL('xgb', this.points, this.models, taskType);
      this.drawXGBLossPlot();

      document.getElementById('xgb-iteration').textContent = this.models.xgb.iteration;
      document.getElementById('xgb-tree-count').textContent = this.models.xgb.trees.length;

      document.getElementById('xgb-metrics').style.display = 'block';
      document.getElementById('xgb-tree-viz').style.display = 'block';
      if (taskType === 'regression') {
        document.getElementById('xgb-reg-metrics').style.display = 'grid';
        document.getElementById('xgb-cls-metrics').style.display = 'none';
      } else {
        document.getElementById('xgb-reg-metrics').style.display = 'none';
        document.getElementById('xgb-cls-metrics').style.display = 'grid';
      }

      if (this.xgboost.autoRunning) {
        setTimeout(() => this.nextBoostXGB(), 100);
      }
    } catch (e) {
      window.showToast(e.message, 'error');
      this.xgboost.stopAutoRun();
    }
  }

  autoXGB() {
    if (this.xgboost.autoRunning) {
      this.xgboost.stopAutoRun();
      document.getElementById('auto-xgb').textContent = 'Auto';
      document.getElementById('auto-xgb').classList.remove('warning');
    } else {
      if (!this.models.xgb.fitted) {
        window.showToast('Click "Initialize" before stepping.', 'warn');
        return;
      }
      this.xgboost.startAutoRun();
      document.getElementById('auto-xgb').textContent = 'Stop';
      document.getElementById('auto-xgb').classList.add('warning');
      this.nextBoostXGB();
    }
  }

  resetXGB() {
    this.xgboost.reset();
    this.models.xgb = { fitted: false, trees: [], basePrediction: 0, iteration: 0 };
    document.getElementById('next-boost-xgb').disabled = true;
    document.getElementById('xgb-iteration').textContent = '0';
    document.getElementById('xgb-tree-count').textContent = '0';
    document.getElementById('xgb-steps').style.display = 'none';
    document.getElementById('xgb-metrics').style.display = 'none';
    document.getElementById('xgb-tree-viz').style.display = 'none';
    document.getElementById('auto-xgb').textContent = 'Auto';
    document.getElementById('auto-xgb').classList.remove('warning');
    this.canvas.drawNL('xgb', this.points, this.models, this.nlTaskType.xgb);
  }

  drawXGBLossPlot() {
    if (this.models.xgb.lossHistory && this.models.xgb.lossHistory.length > 0) {
      this.canvas.drawLossPlot(this.models.xgb.lossHistory, 'xgb-loss-plot');
    }
  }

  // Neural Network Methods
  initNN() {
    try {
      const hiddenSizesStr = document.getElementById('nn-hidden').value;
      const hiddenSizes = hiddenSizesStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0);
      if (hiddenSizes.length === 0) {
        window.showToast('Hidden layer sizes must be a comma-separated list of positive integers (e.g. "8,4").', 'warn');
        return;
      }

      const activation = document.getElementById('nn-activation').value;
      const taskType = this.nlTaskType.nn;

      const model = this.neuralNetwork.initialize(this.getTrainPoints(), taskType, hiddenSizes, activation);
      if (!model) return; // Toast already shown by solver
      this.models.nn = model;
      document.getElementById('next-epoch-nn').disabled = false;
      document.getElementById('nn-epoch').textContent = '0';
      document.getElementById('nn-params').textContent = this.models.nn.totalParams;

      // Show initial explanation with formulas
      const stepsDisplay = document.getElementById('nn-step-display');
      if (stepsDisplay && this.models.nn.initialExplanation) {
        stepsDisplay.innerHTML = this.models.nn.initialExplanation;
        document.getElementById('nn-steps').style.display = 'block';
        // Render inline formulas
        AppUtils.kRenderInlineFormulas(stepsDisplay);
      }

      this.canvas.drawNL('nn', this.points, this.models, taskType);
      this.drawNNLossPlot();
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  nextEpochNN() {
    try {
      let lr = parseFloat(document.getElementById('nn-lr').value);
      let lambda = parseFloat(document.getElementById('nn-lambda').value);
      let dropout = parseFloat(document.getElementById('nn-dropout').value);
      let maxEpochs = parseInt(document.getElementById('nn-max-epochs').value);
      // Validate/clamp hyperparameters to a sane range
      if (!isFinite(lr) || lr <= 0) lr = 1e-6;
      lr = Math.min(Math.max(lr, 1e-6), 10);
      if (!isFinite(lambda) || lambda < 0) lambda = 0;
      if (!isFinite(dropout) || dropout < 0) dropout = 0;
      if (dropout >= 1) dropout = 0.99;
      if (!isFinite(maxEpochs) || maxEpochs < 1) maxEpochs = 1;
      maxEpochs = Math.min(maxEpochs, 10000);
      const taskType = this.nlTaskType.nn;

      // If this is the first epoch, add a separator after the initial explanation
      const stepsDisplay = document.getElementById('nn-step-display');
      if (this.models.nn.epoch === 0 && stepsDisplay) {
        stepsDisplay.innerHTML += '<br><strong>Training Epochs:</strong><br>';
      }

      const train = this.getTrainPoints();
      this.models.nn = this.neuralNetwork.trainEpoch(train, this.models.nn, taskType, lr, lambda, dropout);

      // Append step information
      if (stepsDisplay && this.models.nn.currentStepDisplay) {
        stepsDisplay.innerHTML += this.models.nn.currentStepDisplay;
        // Render inline formulas if any
        AppUtils.kRenderInlineFormulas(stepsDisplay);
      }

      this.neuralNetwork.updateDisplay(train, this.models.nn, taskType);
      this.canvas.drawNL('nn', this.points, this.models, taskType);
      this.drawNNLossPlot();

      document.getElementById('nn-epoch').textContent = this.models.nn.epoch;

      document.getElementById('nn-metrics').style.display = 'block';
      if (taskType === 'regression') {
        document.getElementById('nn-reg-metrics').style.display = 'grid';
        document.getElementById('nn-cls-metrics').style.display = 'none';
      } else {
        document.getElementById('nn-reg-metrics').style.display = 'none';
        document.getElementById('nn-cls-metrics').style.display = 'grid';
      }

      // Stop if max epochs reached
      if (this.neuralNetwork.autoRunning && this.models.nn.epoch >= maxEpochs) {
        this.neuralNetwork.stopAutoRun();
        document.getElementById('auto-nn').textContent = 'Auto';
        document.getElementById('auto-nn').classList.remove('warning');
      } else if (this.neuralNetwork.autoRunning) {
        setTimeout(() => this.nextEpochNN(), 100);
      }
    } catch (e) {
      window.showToast(e.message, 'error');
      this.neuralNetwork.stopAutoRun();
    }
  }

  autoNN() {
    if (this.neuralNetwork.autoRunning) {
      this.neuralNetwork.stopAutoRun();
      document.getElementById('auto-nn').textContent = 'Auto';
      document.getElementById('auto-nn').classList.remove('warning');
    } else {
      if (!this.models.nn.fitted) {
        window.showToast('Click "Initialize" before stepping.', 'warn');
        return;
      }
      this.neuralNetwork.startAutoRun();
      document.getElementById('auto-nn').textContent = 'Stop';
      document.getElementById('auto-nn').classList.add('warning');
      this.nextEpochNN();
    }
  }

  resetNN() {
    this.neuralNetwork.reset();
    this.models.nn = { fitted: false, weights: [], biases: [], epoch: 0 };
    document.getElementById('next-epoch-nn').disabled = true;
    document.getElementById('nn-epoch').textContent = '0';
    document.getElementById('nn-params').textContent = '0';
    document.getElementById('nn-steps').style.display = 'none';
    document.getElementById('nn-metrics').style.display = 'none';
    document.getElementById('auto-nn').textContent = 'Auto';
    document.getElementById('auto-nn').classList.remove('warning');
    this.canvas.drawNL('nn', this.points, this.models, this.nlTaskType.nn);
  }

  drawNNLossPlot() {
    if (this.models.nn.lossHistory && this.models.nn.lossHistory.length > 0) {
      this.canvas.drawLossPlot(this.models.nn.lossHistory, 'nn-loss-plot');
    }
  }

  // ---- RNN ----
  initRNN() {
    const train = this.getTrainPoints();
    const hidden = parseInt(document.getElementById('rnn-hidden').value) || 4;
    const seed = parseInt(document.getElementById('rnn-seed').value) || 42;
    const ok = this.rnn.initialize(train, hidden, seed);
    if (!ok) return;
    this.models.rnn = { fitted: true, predictions: [] };
    document.getElementById('next-step-rnn').disabled = false;
    document.getElementById('run-rnn').disabled = false;
    const trainBtn = document.getElementById('train-rnn');
    if (trainBtn) trainBtn.disabled = false;
    document.getElementById('rnn-step-count').textContent = '0';
    document.getElementById('rnn-step-total').textContent = String(this.rnn.model.ys.length - 1);
    document.getElementById('rnn-mse').textContent = '—';
    this.rnn.renderWeights();
    this.rnn.renderHistory();
    this.canvas.drawNL('rnn', train, this.models, 'regression');
  }

  nextStepRNN() {
    if (!this.rnn.model) { this.initRNN(); if (!this.rnn.model) return; }
    const record = this.rnn.nextStep();
    if (!record) {
      window.showToast('End of sequence reached. Click Reset to start over.', 'info');
      document.getElementById('next-step-rnn').disabled = true;
      return;
    }
    this._refreshRNN(record);
  }

  runRNN() {
    if (!this.rnn.model) { this.initRNN(); if (!this.rnn.model) return; }
    this.rnn.runAll();
    const last = this.rnn.history[this.rnn.history.length - 1];
    this._refreshRNN(last);
    document.getElementById('next-step-rnn').disabled = true;
  }

  _refreshRNN(record) {
    document.getElementById('rnn-step-count').textContent = String(this.rnn.t);
    this.rnn.renderStep(record);
    this.rnn.renderHistory();
    this.rnn.renderMetrics();
    this.models.rnn = { fitted: true, predictions: this.rnn.predictionMarkers() };
    this.canvas.drawNL('rnn', this.getTrainPoints(), this.models, 'regression');
  }

  resetRNN() {
    this.rnn.reset();
    this.models.rnn = { fitted: false, predictions: [] };
    document.getElementById('next-step-rnn').disabled = true;
    document.getElementById('run-rnn').disabled = true;
    const trainBtn = document.getElementById('train-rnn');
    if (trainBtn) trainBtn.disabled = true;
    document.getElementById('rnn-step-count').textContent = '0';
    document.getElementById('rnn-step-total').textContent = '0';
    document.getElementById('rnn-mse').textContent = '—';
    ['rnn-weights', 'rnn-step-detail', 'rnn-history'].forEach(id => {
      const el = document.getElementById(id);
      if (el) { el.innerHTML = ''; if (el.parentElement) el.parentElement.style.display = 'none'; }
    });
    const lossWrap = document.getElementById('rnn-loss-wrap');
    if (lossWrap) lossWrap.style.display = 'none';
    this.canvas.drawNL('rnn', this.points, this.models, 'regression');
  }

  trainRNN() {
    if (!this.rnn.model) { this.initRNN(); if (!this.rnn.model) return; }
    let lr = parseFloat(document.getElementById('rnn-lr').value);
    let epochs = parseInt(document.getElementById('rnn-epochs').value);
    if (!Number.isFinite(lr) || lr <= 0) lr = 0.05;
    lr = Math.min(Math.max(lr, 1e-4), 1);
    if (!Number.isFinite(epochs) || epochs < 1) epochs = 1;
    epochs = Math.min(epochs, 2000);
    const btn = document.getElementById('train-rnn');
    if (btn) { btn.disabled = true; btn.textContent = 'Training…'; }
    // Defer so the UI updates before the heavy synchronous work.
    setTimeout(() => {
      try {
        const lossHistory = this.rnn.train(epochs, lr);
        // Pull the freshly-recomputed history (re-forwarded by train()).
        const last = this.rnn.history[this.rnn.history.length - 1];
        this._refreshRNN(last);
        this.models.rnn.lossHistory = lossHistory;
        const wrap = document.getElementById('rnn-loss-wrap');
        if (wrap) wrap.style.display = 'block';
        this.canvas.drawLossPlot(lossHistory, 'rnn-loss-plot');
        const final = lossHistory[lossHistory.length - 1];
        const initial = lossHistory[0];
        window.showToast(`Trained ${epochs} epochs: MSE ${initial.toFixed(3)} → ${final.toFixed(3)}.`, 'info', 4000);
      } catch (e) {
        window.showToast('Training failed: ' + e.message, 'error');
      } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'Train'; }
      }
    }, 30);
  }

  // ---- LSTM ----
  initLSTM() {
    const train = this.getTrainPoints();
    const hidden = parseInt(document.getElementById('lstm-hidden').value) || 4;
    const seed = parseInt(document.getElementById('lstm-seed').value) || 42;
    const ok = this.lstm.initialize(train, hidden, seed);
    if (!ok) return;
    this.models.lstm = { fitted: true, predictions: [] };
    document.getElementById('next-step-lstm').disabled = false;
    document.getElementById('run-lstm').disabled = false;
    const trainBtn = document.getElementById('train-lstm');
    if (trainBtn) trainBtn.disabled = false;
    document.getElementById('lstm-step-count').textContent = '0';
    document.getElementById('lstm-step-total').textContent = String(this.lstm.model.ys.length - 1);
    document.getElementById('lstm-mse').textContent = '—';
    this.lstm.renderWeights();
    this.lstm.renderHistory();
    this.canvas.drawNL('lstm', train, this.models, 'regression');
  }

  nextStepLSTM() {
    if (!this.lstm.model) { this.initLSTM(); if (!this.lstm.model) return; }
    const record = this.lstm.nextStep();
    if (!record) {
      window.showToast('End of sequence reached. Click Reset to start over.', 'info');
      document.getElementById('next-step-lstm').disabled = true;
      return;
    }
    this._refreshLSTM(record);
  }

  runLSTM() {
    if (!this.lstm.model) { this.initLSTM(); if (!this.lstm.model) return; }
    this.lstm.runAll();
    const last = this.lstm.history[this.lstm.history.length - 1];
    this._refreshLSTM(last);
    document.getElementById('next-step-lstm').disabled = true;
  }

  _refreshLSTM(record) {
    document.getElementById('lstm-step-count').textContent = String(this.lstm.t);
    this.lstm.renderStep(record);
    this.lstm.renderHistory();
    this.lstm.renderMetrics();
    this.models.lstm = { fitted: true, predictions: this.lstm.predictionMarkers() };
    this.canvas.drawNL('lstm', this.getTrainPoints(), this.models, 'regression');
  }

  resetLSTM() {
    this.lstm.reset();
    this.models.lstm = { fitted: false, predictions: [] };
    document.getElementById('next-step-lstm').disabled = true;
    document.getElementById('run-lstm').disabled = true;
    const trainBtn = document.getElementById('train-lstm');
    if (trainBtn) trainBtn.disabled = true;
    document.getElementById('lstm-step-count').textContent = '0';
    document.getElementById('lstm-step-total').textContent = '0';
    document.getElementById('lstm-mse').textContent = '—';
    ['lstm-weights', 'lstm-step-detail', 'lstm-history'].forEach(id => {
      const el = document.getElementById(id);
      if (el) { el.innerHTML = ''; if (el.parentElement) el.parentElement.style.display = 'none'; }
    });
    const lossWrap = document.getElementById('lstm-loss-wrap');
    if (lossWrap) lossWrap.style.display = 'none';
    this.canvas.drawNL('lstm', this.points, this.models, 'regression');
  }

  trainLSTM() {
    if (!this.lstm.model) { this.initLSTM(); if (!this.lstm.model) return; }
    let lr = parseFloat(document.getElementById('lstm-lr').value);
    let epochs = parseInt(document.getElementById('lstm-epochs').value);
    if (!Number.isFinite(lr) || lr <= 0) lr = 0.05;
    lr = Math.min(Math.max(lr, 1e-4), 1);
    if (!Number.isFinite(epochs) || epochs < 1) epochs = 1;
    epochs = Math.min(epochs, 2000);
    const btn = document.getElementById('train-lstm');
    if (btn) { btn.disabled = true; btn.textContent = 'Training…'; }
    setTimeout(() => {
      try {
        const lossHistory = this.lstm.train(epochs, lr);
        const last = this.lstm.history[this.lstm.history.length - 1];
        this._refreshLSTM(last);
        this.models.lstm.lossHistory = lossHistory;
        const wrap = document.getElementById('lstm-loss-wrap');
        if (wrap) wrap.style.display = 'block';
        this.canvas.drawLossPlot(lossHistory, 'lstm-loss-plot');
        const final = lossHistory[lossHistory.length - 1];
        const initial = lossHistory[0];
        window.showToast(`Trained ${epochs} epochs: MSE ${initial.toFixed(3)} → ${final.toFixed(3)}.`, 'info', 4000);
      } catch (e) {
        window.showToast('Training failed: ' + e.message, 'error');
      } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'Train'; }
      }
    }, 30);
  }

  addPoint(point) {
    this.points.push(point);
    this.recomputeSplit();
    this.redrawCanvas();
  }

  removePoint(index) {
    this.points.splice(index, 1);
    this.redrawCanvas();
  }

  updatePoint(index, point) {
    this.points[index] = point;
    this.redrawCanvas();
  }

  redrawCanvas() {
    if (this.activeTab === 'linear') {
      this.canvas.draw(this.activeRegTab, this.points, this.models, this.showResiduals);
      if (this.activeRegTab === 'gd') {
        this.canvas.drawLossPlot(this.gradientDescent.lossHistory);
      } else if (this.activeRegTab === 'logistic') {
        this.drawLRLossPlot();
      }
    } else if (this.activeTab === 'nonlinear') {
      const taskType = this.nlTaskType[this.activeNLTab];
      this.canvas.drawNL(this.activeNLTab, this.points, this.models, taskType);
      if (this.activeNLTab === 'xgb' && this.models.xgb.fitted) {
        this.drawXGBLossPlot();
      } else if (this.activeNLTab === 'nn' && this.models.nn.fitted) {
        this.drawNNLossPlot();
      }
    } else if (this.activeTab === 'clustering') {
      this.drawClusteringCanvas();
    }

    // Update point count
    const countEl = document.getElementById('point-count');
    if (countEl) countEl.textContent = this.points.length;
  }

  getPoints() { return this.points; }

  setLogisticGroup(groupIdx) {
    this.activeLogisticGroup = groupIdx;
    // Update button styles
    document.getElementById('lr-group0-btn').className = groupIdx === 0 ? 'warning' : '';
    document.getElementById('lr-group1-btn').className = groupIdx === 1 ? 'success' : '';
  }

  getActiveLogisticGroup() {
    return this.activeLogisticGroup;
  }

  solveOLS() {
    try {
      const train = this.getTrainPoints();
      const test = this.getTestPoints();
      this.models.ols = this.olsSolver.solve(train);
      this.olsSolver.displaySteps(train, this.models.ols);
      this.olsSolver.updateDisplay(train, this.models.ols, test);
      this.canvas.draw(this.activeRegTab, this.points, this.models, this.showResiduals);
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  initializeGD() {
    try {
      this.models.gd = this.gradientDescent.initialize(this.getTrainPoints());
      this.canvas.draw(this.activeRegTab, this.points, this.models, this.showResiduals);
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  nextGDStep() {
    this.models.gd = this.gradientDescent.executeStep(this.getTrainPoints(), this.models.gd);
    this.canvas.draw(this.activeRegTab, this.points, this.models, this.showResiduals);
    this.canvas.drawLossPlot(this.gradientDescent.lossHistory);
  }

  autoRunGD() {
    if (!this.gradientDescent.initialized) {
      this.initializeGD();
    }

    this.gradientDescent.startAutoRun(
      this.getTrainPoints(),
      () => this.models.gd, // Function to get current model
      (newModel) => {       // Callback to update model
        this.models.gd = newModel;
        this.canvas.draw(this.activeRegTab, this.points, this.models, this.showResiduals);
        this.canvas.drawLossPlot(this.gradientDescent.lossHistory);
      }
    );
  }

  resetGD() {
    this.gradientDescent.stopAutoRun();
    this.gradientDescent.reset();
    this.models.gd = {m: 0, b: 0, fitted: false};
    document.getElementById('next-step').disabled = true;
    for (let i = 0; i <= 4; i++) {
      const el = document.getElementById(`step-${i}`);
      if (el) el.style.display = 'none';
    }
    this.gradientDescent.updateStepDisplay();
    this.canvas.draw(this.activeRegTab, this.points, this.models, this.showResiduals);
    this.canvas.drawLossPlot(this.gradientDescent.lossHistory);
  }

  calculateManual() {
    try {
      this.models.manual = this.manualCalculator.calculate(this.getTrainPoints());
      this.canvas.draw(this.activeRegTab, this.points, this.models, this.showResiduals);
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  clear() {
    this.points = [];
    this.models = {
      ols: {m:0, b:0, fitted:false},
      gd: {m:0, b:0, fitted:false},
      manual: {m:0, b:0, fitted:false},
      logistic: {m:0, b:0, fitted:false},
      dt: {fitted: false, tree: null},
      rf: {fitted: false, trees: [], oobIndices: []},
      xgb: {fitted: false, trees: [], basePrediction: 0, iteration: 0}
    };
    this.gradientDescent.stopAutoRun();
    this.gradientDescent.reset();
    this.logisticRegression.stopAutoRun();
    this.logisticRegression.reset();

    const nextStep = document.getElementById('next-step');
    if (nextStep) nextStep.disabled = true;
    const nextStepLR = document.getElementById('next-step-lr');
    if (nextStepLR) nextStepLR.disabled = true;

    const olsCalc = document.getElementById('ols-calculations');
    if (olsCalc) olsCalc.style.display = 'none';
    const manualCalc = document.getElementById('manual-calculations');
    if (manualCalc) manualCalc.style.display = 'none';

    for (let i = 0; i <= 4; i++) {
      const el = document.getElementById(`step-${i}`);
      if (el) el.style.display = 'none';
      const elLR = document.getElementById(`lr-step-${i}`);
      if (elLR) elLR.style.display = 'none';
    }

    this.gradientDescent.updateStepDisplay();
    this.logisticRegression.updateStepDisplay();

    this.redrawCanvas();

    const accEl = document.getElementById('lr-accuracy');
    if (accEl) accEl.textContent = '—';
  }

  generateSample() {
    this.points = [];
    const n = 30;

    // Determine if we need classification or regression data
    const isLinearClassification = this.activeTab === 'linear' && this.activeRegTab === 'logistic';
    const isNonlinearClassification = this.activeTab === 'nonlinear' &&
      this.nlTaskType[this.activeNLTab] === 'classification';
    const isClassification = isLinearClassification || isNonlinearClassification;

    if (isClassification) {
      // Generate classification data with clear x-based separation
      for (let i = 0; i < n; i++) {
        const x = i * 10 / (n - 1);
        const label = x > 5 ? 1 : 0; // Separate by x position
        const centerY = label === 1 ? 6 + Math.random() * 3 : 2 + Math.random() * 3;
        const y = centerY + (Math.random() - 0.5) * 1;
        this.points.push({
          x: Math.max(0, Math.min(10, x)),
          y: Math.max(0, Math.min(10, y)),
          label
        });
      }
    } else {
      // Generate regression data
      for (let i = 0; i < n; i++) {
        const x = i * 10 / (n - 1);
        const y = 0.7 * x + 1.5 + (Math.random() - 0.5) * 2.0;
        this.points.push({
          x: Math.max(0, Math.min(10, x)),
          y: Math.max(0, Math.min(10, y))
        });
      }
    }

    this.redrawCanvas();
  }

  // Logistic Regression Methods
  initializeLR() {
    try {
      this.models.logistic = this.logisticRegression.initialize(this.getTrainPoints());
      this.canvas.draw(this.activeRegTab, this.points, this.models, this.showResiduals);
      this.updateLRMetrics();
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  nextLRStep() {
    this.models.logistic = this.logisticRegression.executeStep(this.getTrainPoints(), this.models.logistic);
    this.canvas.draw(this.activeRegTab, this.points, this.models, this.showResiduals);
    this.drawLRLossPlot();
    this.updateLRMetrics();
  }

  autoRunLR() {
    if (!this.logisticRegression.initialized) {
      this.initializeLR();
    }

    this.logisticRegression.startAutoRun(
      this.getTrainPoints(),
      () => this.models.logistic,
      (newModel) => {
        this.models.logistic = newModel;
        this.canvas.draw(this.activeRegTab, this.points, this.models, this.showResiduals);
        this.drawLRLossPlot();
        this.updateLRMetrics();
      }
    );
  }

  resetLR() {
    this.logisticRegression.stopAutoRun();
    this.logisticRegression.reset();
    this.models.logistic = {m: 0, b: 0, fitted: false};
    const nextBtn = document.getElementById('next-step-lr');
    if (nextBtn) nextBtn.disabled = true;
    for (let i = 0; i <= 4; i++) {
      const el = document.getElementById(`lr-step-${i}`);
      if (el) el.style.display = 'none';
    }
    this.logisticRegression.updateStepDisplay();
    this.canvas.draw(this.activeRegTab, this.points, this.models, this.showResiduals);
    this.drawLRLossPlot();
    const accEl = document.getElementById('lr-accuracy');
    if (accEl) accEl.textContent = '—';
  }

  updateLRMetrics(threshold = 0.5) {
    if (!this.models.logistic.fitted || this.points.length === 0) return;

    // Metrics computed on training set; Youden threshold likewise computed on train only
    // to avoid leakage. Test points are visually rendered as rings on the canvas.
    const train = this.getTrainPoints();
    const metrics = this.logisticRegression.calculateMetrics(train, this.models.logistic, threshold);
    if (!metrics) return;

    // Update threshold display
    document.getElementById('lr-threshold-val').textContent = threshold.toFixed(2);

    // Update confusion matrix
    document.getElementById('cm-tp').textContent = metrics.tp;
    document.getElementById('cm-fp').textContent = metrics.fp;
    document.getElementById('cm-tn').textContent = metrics.tn;
    document.getElementById('cm-fn').textContent = metrics.fn;
    document.getElementById('lr-confusion-matrix').style.display = 'block';

    // Update metrics
    document.getElementById('lr-accuracy').textContent = (metrics.accuracy * 100).toFixed(2) + '%';
    document.getElementById('lr-precision').textContent = (metrics.precision * 100).toFixed(2) + '%';
    document.getElementById('lr-recall').textContent = (metrics.recall * 100).toFixed(2) + '%';
    document.getElementById('lr-specificity').textContent = (metrics.specificity * 100).toFixed(2) + '%';
    document.getElementById('lr-f1').textContent = metrics.f1.toFixed(4);

    // Calculate and display Youden index (on train only — avoid leakage)
    const youdenThreshold = this.logisticRegression.calculateYoudenThreshold(train, this.models.logistic);
    document.getElementById('lr-youden').textContent = youdenThreshold.toFixed(2);
  }

  useYoudenThreshold() {
    if (!this.models.logistic.fitted || this.points.length === 0) return;
    const youdenThreshold = this.logisticRegression.calculateYoudenThreshold(this.points, this.models.logistic);
    document.getElementById('lr-threshold').value = youdenThreshold;
    this.updateLRMetrics(youdenThreshold);
  }

  drawLRLossPlot() {
    const canvas = document.getElementById('lr-loss-plot');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const lossHistory = this.logisticRegression.lossHistory;
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

  // Statistics Methods
  setTTestGroup(groupIdx) {
    this.statData.ttest.activeGroup = groupIdx;
    // Update button styles
    document.getElementById('ttest-group1-btn').className = groupIdx === 0 ? 'success' : '';
    document.getElementById('ttest-group2-btn').className = groupIdx === 1 ? 'success' : '';
  }

  setAnovaGroup(groupIdx) {
    this.statData.anova.activeGroup = groupIdx;
    // Update button styles
    for (let i = 1; i <= 5; i++) {
      document.getElementById(`anova-g${i}-btn`).className = (groupIdx === i - 1) ? 'success' : '';
    }
  }

  clearStatData(type) {
    if (type === 'correlation') {
      this.statData.correlation = [];
      this.canvas.drawStatisticsPlot('plot-correlation', { group1: [], group2: [] }, 'correlation');
      document.getElementById('corr-point-count').textContent = '0';
      document.getElementById('corr-results').style.display = 'none';
    } else if (type === 'ttest') {
      this.statData.ttest = { group1: [], group2: [] };
      this.canvas.drawStatisticsPlot('plot-ttest', { group1: [], group2: [], mean1: 0, mean2: 0 }, 'ttest');
      document.getElementById('ttest-n1').textContent = '0';
      document.getElementById('ttest-n2').textContent = '0';
      document.getElementById('ttest-results').style.display = 'none';
    } else if (type === 'anova') {
      this.statData.anova.groups = [[], [], [], [], []];
      this.canvas.drawStatisticsPlot('plot-anova', { groups: [], groupMeans: [] }, 'anova');
      for (let i = 1; i <= 5; i++) {
        document.getElementById(`anova-g${i}`).textContent = `G${i}: 0`;
      }
      document.getElementById('anova-results').style.display = 'none';
    }
  }

  sampleStatData(type) {
    if (type === 'correlation') {
      this.statData.correlation = [];
      for (let i = 0; i < 20; i++) {
        const x = Math.random() * 10;
        const y = x * 0.7 + Math.random() * 3;
        this.statData.correlation.push({ x, y });
      }
      document.getElementById('corr-point-count').textContent = this.statData.correlation.length;
      const group1 = this.statData.correlation.map(p => p.x);
      const group2 = this.statData.correlation.map(p => p.y);
      this.canvas.drawStatisticsPlot('plot-correlation', { group1, group2 }, 'correlation');
    } else if (type === 'ttest') {
      this.statData.ttest.group1 = Array.from({length: 15}, () => 5 + Math.random() * 3);
      this.statData.ttest.group2 = Array.from({length: 15}, () => 8 + Math.random() * 3);
      document.getElementById('ttest-n1').textContent = this.statData.ttest.group1.length;
      document.getElementById('ttest-n2').textContent = this.statData.ttest.group2.length;
      const mean1 = this.statData.ttest.group1.reduce((s,v) => s+v, 0) / this.statData.ttest.group1.length;
      const mean2 = this.statData.ttest.group2.reduce((s,v) => s+v, 0) / this.statData.ttest.group2.length;
      this.canvas.drawStatisticsPlot('plot-ttest', {...this.statData.ttest, mean1, mean2 }, 'ttest');
    } else if (type === 'anova') {
      for (let g = 0; g < 3; g++) {
        this.statData.anova.groups[g] = Array.from({length: 10}, () => (g * 3 + 5) + Math.random() * 2);
      }
      for (let i = 1; i <= 5; i++) {
        document.getElementById(`anova-g${i}`).textContent = `G${i}: ${this.statData.anova.groups[i-1].length}`;
      }
      const groupMeans = this.statData.anova.groups.map(g => g.length > 0 ? g.reduce((s,v)=>s+v,0)/g.length : 0);
      this.canvas.drawStatisticsPlot('plot-anova', { groups: this.statData.anova.groups.filter(g => g.length > 0), groupMeans }, 'anova');
    }
  }

  calculateCorrelation() {
    try {
      if (this.statData.correlation.length < 2) {
        window.showToast('Need at least 2 points. Click the canvas to add some, or use Sample Data.', 'warn');
        return;
      }

      const group1 = this.statData.correlation.map(p => p.x);
      const group2 = this.statData.correlation.map(p => p.y);
      const result = this.statisticsCalculator.calculateCorrelation(group1, group2);

      // Calculate covariance and standard deviations for detailed steps
      const n = result.n;
      let cov = 0, var1 = 0, var2 = 0;
      for (let i = 0; i < n; i++) {
        const d1 = group1[i] - result.mean1;
        const d2 = group2[i] - result.mean2;
        cov += d1 * d2;
        var1 += d1 * d1;
        var2 += d2 * d2;
      }
      cov /= (n - 1);
      const std1 = Math.sqrt(var1 / (n - 1));
      const std2 = Math.sqrt(var2 / (n - 1));

      // Display formulas
      AppUtils.kRender(document.getElementById('corr-formula-tex'),
        `r = \\frac{\\text{Cov}(x,y)}{s_x s_y}`, true);
      AppUtils.kRender(document.getElementById('corr-cov-formula-tex'),
        `\\text{Cov}(x,y) = \\frac{1}{n-1}\\sum (x_i - \\bar{x})(y_i - \\bar{y})`, true);
      AppUtils.kRender(document.getElementById('corr-std-formula-tex'),
        `s_x = \\sqrt{\\frac{1}{n-1}\\sum (x_i - \\bar{x})^2}, \\quad s_y = \\sqrt{\\frac{1}{n-1}\\sum (y_i - \\bar{y})^2}`, true);

      // Display steps
      AppUtils.kRender(document.getElementById('corr-n-tex'), `n = ${n}`);
      AppUtils.kRender(document.getElementById('corr-mean1-tex'), `\\bar{x} = ${result.mean1.toFixed(4)}`);
      AppUtils.kRender(document.getElementById('corr-mean2-tex'), `\\bar{y} = ${result.mean2.toFixed(4)}`);
      AppUtils.kRender(document.getElementById('corr-cov-tex'), `\\text{Cov}(x,y) = ${cov.toFixed(4)}`);
      AppUtils.kRender(document.getElementById('corr-std1-tex'), `s_x = ${std1.toFixed(4)}`);
      AppUtils.kRender(document.getElementById('corr-std2-tex'), `s_y = ${std2.toFixed(4)}`);

      document.getElementById('corr-r-val').textContent = result.r.toFixed(6);
      document.getElementById('corr-results').style.display = 'block';
      this.canvas.drawStatisticsPlot('plot-correlation', { group1, group2 }, 'correlation');
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  calculateTTest() {
    try {
      const { group1, group2 } = this.statData.ttest;
      if (group1.length < 2 || group2.length < 2) {
        window.showToast('Each group needs at least 2 observations. Click canvas to add or use Sample Data.', 'warn');
        return;
      }

      const result = this.statisticsCalculator.calculateTTest(group1, group2);

      // Display formulas
      AppUtils.kRender(document.getElementById('ttest-formula-tex'),
        `t = \\frac{\\bar{x}_1 - \\bar{x}_2}{SE}`, true);
      AppUtils.kRender(document.getElementById('ttest-pooled-formula-tex'),
        `s_p^2 = \\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}`, true);
      AppUtils.kRender(document.getElementById('ttest-se-formula-tex'),
        `SE = \\sqrt{s_p^2(\\frac{1}{n_1} + \\frac{1}{n_2})}`, true);

      // Display steps
      AppUtils.kRender(document.getElementById('ttest-n-tex'), `n_1 = ${result.n1}, \\quad n_2 = ${result.n2}`);
      AppUtils.kRender(document.getElementById('ttest-mean1-tex'), `\\bar{x}_1 = ${result.mean1.toFixed(4)}`);
      AppUtils.kRender(document.getElementById('ttest-mean2-tex'), `\\bar{x}_2 = ${result.mean2.toFixed(4)}`);
      AppUtils.kRender(document.getElementById('ttest-var1-tex'), `s_1^2 = ${result.var1.toFixed(4)}`);
      AppUtils.kRender(document.getElementById('ttest-var2-tex'), `s_2^2 = ${result.var2.toFixed(4)}`);

      const pooledVar = ((result.n1 - 1) * result.var1 + (result.n2 - 1) * result.var2) / (result.n1 + result.n2 - 2);
      AppUtils.kRender(document.getElementById('ttest-pooled-tex'), `s_p^2 = ${pooledVar.toFixed(4)}`);
      AppUtils.kRender(document.getElementById('ttest-se-tex'), `SE = ${result.se.toFixed(4)}`);
      AppUtils.kRender(document.getElementById('ttest-df-tex'), `df = ${result.df}`);

      // Calculate p-value based on test type
      const testType = this.statData.ttest.testType;
      let pValue;
      let hypothesisText = '';

      if (testType === 'two-tailed') {
        pValue = result.pValue; // Already two-tailed from calculator
        hypothesisText = 'μ₁ ≠ μ₂';
      } else if (testType === 'left') {
        // One-tailed left: H₁: μ₁ < μ₂ (or t < 0)
        pValue = this.statisticsCalculator.tCDF(-Math.abs(result.t), result.df);
        if (result.t < 0) {
          pValue = 1 - this.statisticsCalculator.tCDF(Math.abs(result.t), result.df);
        }
        hypothesisText = 'μ₁ < μ₂';
      } else { // right
        // One-tailed right: H₁: μ₁ > μ₂ (or t > 0)
        pValue = 1 - this.statisticsCalculator.tCDF(Math.abs(result.t), result.df);
        if (result.t < 0) {
          pValue = this.statisticsCalculator.tCDF(-Math.abs(result.t), result.df);
        }
        hypothesisText = 'μ₁ > μ₂';
      }

      document.getElementById('ttest-t-val').textContent = result.t.toFixed(6);
      document.getElementById('ttest-df-val').textContent = result.df;
      document.getElementById('ttest-t-abs').textContent = Math.abs(result.t).toFixed(4);
      document.getElementById('ttest-pval').textContent = pValue.toFixed(6);

      // P-value interpretation
      let pvalText = '';
      const testTypeLabel = testType === 'two-tailed' ? 'Two-tailed' :
                           testType === 'left' ? 'One-tailed (left)' : 'One-tailed (right)';

      if (pValue > 0.05) {
        pvalText = `${testTypeLabel} test: Not significant at α=0.05 (p = ${pValue.toFixed(4)} > 0.05). Fail to reject H₀. No evidence for H₁: ${hypothesisText}.`;
      } else if (pValue > 0.01) {
        pvalText = `${testTypeLabel} test: Significant at α=0.05 (p = ${pValue.toFixed(4)} < 0.05). Reject H₀. Evidence for H₁: ${hypothesisText}.`;
      } else {
        pvalText = `${testTypeLabel} test: Highly significant at α=0.01 (p = ${pValue.toFixed(4)} < 0.01). Strong evidence for H₁: ${hypothesisText}.`;
      }
      document.getElementById('ttest-pval-text').textContent = pvalText;

      // Draw distribution
      this.drawTDistribution(result.t, result.df, testType);

      document.getElementById('ttest-results').style.display = 'block';
      this.canvas.drawStatisticsPlot('plot-ttest', { group1, group2, mean1: result.mean1, mean2: result.mean2 }, 'ttest');
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  calculateANOVA() {
    try {
      const groups = this.statData.anova.groups.filter(g => g.length > 0);
      if (groups.length < 2) {
        window.showToast('Need at least 2 groups with data. Click canvas and press keys 1-5 to switch group, or use Sample Data.', 'warn');
        return;
      }

      const result = this.statisticsCalculator.calculateANOVA(groups);

      // Display formulas
      AppUtils.kRender(document.getElementById('anova-formula-tex'),
        `F = \\frac{MS_{between}}{MS_{within}} = \\frac{SS_B / df_B}{SS_W / df_W}`, true);
      AppUtils.kRender(document.getElementById('anova-ssb-formula-tex'),
        `SS_B = \\sum_{j=1}^{k} n_j(\\bar{x}_j - \\bar{x}_{grand})^2`, true);
      AppUtils.kRender(document.getElementById('anova-ssw-formula-tex'),
        `SS_W = \\sum_{j=1}^{k} \\sum_{i=1}^{n_j} (x_{ij} - \\bar{x}_j)^2`, true);
      AppUtils.kRender(document.getElementById('anova-df-formula-tex'),
        `df_B = k - 1, \\quad df_W = N - k`, true);

      // Display steps
      AppUtils.kRender(document.getElementById('anova-grand-tex'), `\\bar{x}_{grand} = ${result.grandMean.toFixed(4)}`);
      const meansStr = result.groupMeans.map((m, i) => `\\bar{x}_{${i+1}} = ${m.toFixed(4)}`).join(', \\; ');
      AppUtils.kRender(document.getElementById('anova-means-tex'), meansStr);
      AppUtils.kRender(document.getElementById('anova-ssb-tex'), `SS_B = ${result.ssb.toFixed(4)}`);
      AppUtils.kRender(document.getElementById('anova-ssw-tex'), `SS_W = ${result.ssw.toFixed(4)}`);
      AppUtils.kRender(document.getElementById('anova-dfb-tex'), `df_B = k - 1 = ${result.dfb}`);
      AppUtils.kRender(document.getElementById('anova-dfw-tex'), `df_W = N - k = ${result.dfw}`);
      AppUtils.kRender(document.getElementById('anova-msb-tex'), `MS_B = SS_B / df_B = ${result.msb.toFixed(4)}`);
      AppUtils.kRender(document.getElementById('anova-msw-tex'), `MS_W = SS_W / df_W = ${result.msw.toFixed(4)}`);

      document.getElementById('anova-f-val').textContent = result.f.toFixed(6);
      document.getElementById('anova-df-val').textContent = `${result.dfb}, ${result.dfw}`;
      document.getElementById('anova-pval').textContent = result.pValue.toFixed(6);

      // P-value interpretation
      let pvalText = '';
      if (result.pValue > 0.05) {
        pvalText = `Not significant at α=0.05 (p = ${result.pValue.toFixed(4)} > 0.05). Fail to reject H₀: group means are not significantly different.`;
      } else if (result.pValue > 0.01) {
        pvalText = `Significant at α=0.05 (p = ${result.pValue.toFixed(4)} < 0.05). Reject H₀: at least one group mean differs.`;
      } else {
        pvalText = `Highly significant at α=0.01 (p = ${result.pValue.toFixed(4)} < 0.01). Strong evidence that group means differ.`;
      }
      document.getElementById('anova-pval-text').textContent = pvalText;

      // Draw distribution
      this.drawFDistribution(result.f, result.dfb, result.dfw);

      document.getElementById('anova-results').style.display = 'block';
      this.canvas.drawStatisticsPlot('plot-anova', result, 'anova');
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  drawTDistribution(tStat, df, testType = 'two-tailed') {
    const canvas = document.getElementById('ttest-dist-plot');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;

    // Clear
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, w, h);

    const marginL = 40, marginR = 20, marginT = 20, marginB = 40;
    const plotW = w - marginL - marginR;
    const plotH = h - marginT - marginB;

    // Determine x range based on t-stat and df
    const tAbs = Math.abs(tStat);
    const xMax = Math.max(4, tAbs + 1);
    const xMin = -xMax;

    // Helper: t-distribution PDF
    const tPDF = (x, df) => {
      const num = Math.exp(this.statisticsCalculator.lnGamma((df + 1) / 2));
      const denom = Math.sqrt(df * Math.PI) * Math.exp(this.statisticsCalculator.lnGamma(df / 2));
      return (num / denom) * Math.pow(1 + (x * x) / df, -(df + 1) / 2);
    };

    // Find max y for scaling
    let maxY = tPDF(0, df);

    // Draw axes
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    ctx.strokeRect(marginL, marginT, plotW, plotH);

    // Draw t-distribution curve
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const x = xMin + (i / 200) * (xMax - xMin);
      const y = tPDF(x, df);
      const px = marginL + ((x - xMin) / (xMax - xMin)) * plotW;
      const py = marginT + plotH - (y / maxY) * plotH;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Shade rejection regions based on test type
    ctx.fillStyle = 'rgba(239, 68, 68, 0.3)';

    if (testType === 'two-tailed') {
      // Left tail
      ctx.beginPath();
      const leftCrit = -tAbs;
      for (let i = 0; i <= 100; i++) {
        const x = xMin + (i / 100) * (leftCrit - xMin);
        const y = tPDF(x, df);
        const px = marginL + ((x - xMin) / (xMax - xMin)) * plotW;
        const py = marginT + plotH - (y / maxY) * plotH;
        if (i === 0) {
          ctx.moveTo(px, marginT + plotH);
          ctx.lineTo(px, py);
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.lineTo(marginL + ((leftCrit - xMin) / (xMax - xMin)) * plotW, marginT + plotH);
      ctx.closePath();
      ctx.fill();

      // Right tail
      ctx.beginPath();
      const rightCrit = tAbs;
      for (let i = 0; i <= 100; i++) {
        const x = rightCrit + (i / 100) * (xMax - rightCrit);
        const y = tPDF(x, df);
        const px = marginL + ((x - xMin) / (xMax - xMin)) * plotW;
        const py = marginT + plotH - (y / maxY) * plotH;
        if (i === 0) {
          ctx.moveTo(px, marginT + plotH);
          ctx.lineTo(px, py);
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.lineTo(marginL + plotW, marginT + plotH);
      ctx.closePath();
      ctx.fill();
    } else if (testType === 'left') {
      // Shade left tail only (from -infinity to tStat)
      ctx.beginPath();
      for (let i = 0; i <= 100; i++) {
        const x = xMin + (i / 100) * (tStat - xMin);
        const y = tPDF(x, df);
        const px = marginL + ((x - xMin) / (xMax - xMin)) * plotW;
        const py = marginT + plotH - (y / maxY) * plotH;
        if (i === 0) {
          ctx.moveTo(px, marginT + plotH);
          ctx.lineTo(px, py);
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.lineTo(marginL + ((tStat - xMin) / (xMax - xMin)) * plotW, marginT + plotH);
      ctx.closePath();
      ctx.fill();
    } else { // right
      // Shade right tail only (from tStat to +infinity)
      ctx.beginPath();
      for (let i = 0; i <= 100; i++) {
        const x = tStat + (i / 100) * (xMax - tStat);
        const y = tPDF(x, df);
        const px = marginL + ((x - xMin) / (xMax - xMin)) * plotW;
        const py = marginT + plotH - (y / maxY) * plotH;
        if (i === 0) {
          ctx.moveTo(px, marginT + plotH);
          ctx.lineTo(px, py);
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.lineTo(marginL + plotW, marginT + plotH);
      ctx.closePath();
      ctx.fill();
    }

    // Draw vertical line for observed t-statistic
    ctx.strokeStyle = '#dc2626';
    ctx.lineWidth = 3;
    const tPx = marginL + ((tStat - xMin) / (xMax - xMin)) * plotW;
    ctx.beginPath();
    ctx.moveTo(tPx, marginT);
    ctx.lineTo(tPx, marginT + plotH);
    ctx.stroke();

    // Draw x-axis labels
    ctx.fillStyle = '#374151';
    ctx.font = '11px system-ui';
    ctx.textAlign = 'center';
    for (let x = Math.ceil(xMin); x <= Math.floor(xMax); x++) {
      const px = marginL + ((x - xMin) / (xMax - xMin)) * plotW;
      ctx.fillText(x.toFixed(0), px, h - marginB + 15);
    }

    // Label
    ctx.fillText('t-value', w / 2, h - 5);
  }

  drawFDistribution(fStat, df1, df2) {
    const canvas = document.getElementById('anova-dist-plot');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;

    // Clear
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, w, h);

    const marginL = 40, marginR = 20, marginT = 20, marginB = 40;
    const plotW = w - marginL - marginR;
    const plotH = h - marginT - marginB;

    // Determine x range
    const xMin = 0;
    const xMax = Math.max(6, fStat + 1);

    // Helper: F-distribution PDF approximation
    const fPDF = (x, d1, d2) => {
      if (x <= 0) return 0;
      const a = d1 / 2;
      const b = d2 / 2;
      const num = Math.exp(this.statisticsCalculator.lnGamma(a + b) - this.statisticsCalculator.lnGamma(a) - this.statisticsCalculator.lnGamma(b));
      const pow1 = Math.pow(d1 / d2, a);
      const pow2 = Math.pow(x, a - 1);
      const pow3 = Math.pow(1 + (d1 * x) / d2, -(a + b));
      return num * pow1 * pow2 * pow3;
    };

    // Find max y for scaling
    let maxY = 0;
    for (let i = 0; i <= 200; i++) {
      const x = xMin + (i / 200) * (xMax - xMin);
      const y = fPDF(x, df1, df2);
      if (y > maxY) maxY = y;
    }

    // Draw axes
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    ctx.strokeRect(marginL, marginT, plotW, plotH);

    // Draw F-distribution curve
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const x = xMin + (i / 200) * (xMax - xMin);
      const y = fPDF(x, df1, df2);
      const px = marginL + ((x - xMin) / (xMax - xMin)) * plotW;
      const py = marginT + plotH - (y / maxY) * plotH;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Shade rejection region (right tail)
    ctx.fillStyle = 'rgba(239, 68, 68, 0.3)';
    ctx.beginPath();
    for (let i = 0; i <= 100; i++) {
      const x = fStat + (i / 100) * (xMax - fStat);
      const y = fPDF(x, df1, df2);
      const px = marginL + ((x - xMin) / (xMax - xMin)) * plotW;
      const py = marginT + plotH - (y / maxY) * plotH;
      if (i === 0) {
        ctx.moveTo(px, marginT + plotH);
        ctx.lineTo(px, py);
      } else {
        ctx.lineTo(px, py);
      }
    }
    ctx.lineTo(marginL + plotW, marginT + plotH);
    ctx.closePath();
    ctx.fill();

    // Draw vertical line for observed F-statistic
    ctx.strokeStyle = '#dc2626';
    ctx.lineWidth = 3;
    const fPx = marginL + ((fStat - xMin) / (xMax - xMin)) * plotW;
    ctx.beginPath();
    ctx.moveTo(fPx, marginT);
    ctx.lineTo(fPx, marginT + plotH);
    ctx.stroke();

    // Draw x-axis labels
    ctx.fillStyle = '#374151';
    ctx.font = '11px system-ui';
    ctx.textAlign = 'center';
    for (let x = 0; x <= Math.floor(xMax); x++) {
      const px = marginL + ((x - xMin) / (xMax - xMin)) * plotW;
      ctx.fillText(x.toFixed(0), px, h - marginB + 15);
    }

    // Label
    ctx.fillText('F-value', w / 2, h - 5);
  }

  // ============================================================================
  // RECOMMENDATION SYSTEMS METHODS
  // ============================================================================

  addRecSysUser() {
    try {
      this.recommendationSystem.addUserRow();
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  addRecSysMovie() {
    try {
      this.recommendationSystem.addMovieColumn();
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  resetRecSysData() {
    try {
      this.recommendationSystem.resetToDefault();
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  randomRecSysData() {
    try {
      this.recommendationSystem.generateRandomRatings();
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  addMovieFeature() {
    try {
      this.contentBasedFiltering.addFeature();
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  resetMovieFeatures() {
    try {
      this.contentBasedFiltering.resetFeatures();
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  randomMovieFeatures() {
    try {
      this.contentBasedFiltering.randomizeFeatures();
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  computeContentBased() {
    try {
      this.contentBasedFiltering.computeRecommendations();
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  initCF() {
    try {
      this.collaborativeFiltering.initialize();
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  nextEpochCF() {
    try {
      this.collaborativeFiltering.trainEpoch();
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  autoCF() {
    try {
      if (!this.collaborativeFiltering.initialized) {
        this.collaborativeFiltering.initialize();
      }
      this.collaborativeFiltering.autoRun();
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  resetCF() {
    this.collaborativeFiltering.reset();
    document.getElementById('next-epoch-cf').disabled = true;
    document.getElementById('cf-epoch').textContent = '0';
    document.getElementById('cf-loss').textContent = '—';
    document.getElementById('cf-equations').style.display = 'none';
    document.getElementById('cf-matrices').style.display = 'none';
    document.getElementById('cf-predictions').style.display = 'none';
    document.getElementById('cf-metrics').style.display = 'none';

    // Clear canvases
    const lossCanvas = document.getElementById('cf-loss-plot');
    const lossCtx = lossCanvas.getContext('2d');
    lossCtx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);

    const heatmapCanvas = document.getElementById('cf-heatmap');
    const heatmapCtx = heatmapCanvas.getContext('2d');
    heatmapCtx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
  }

  // ============================================================================
  // CLUSTERING METHODS
  // ============================================================================

  onClusteringCanvasClick(e, canvas) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const px = (e.clientX - rect.left) * scaleX;
    const py = (e.clientY - rect.top) * scaleY;

    if (e.button === 2) {
      // Right-click: remove last point
      if (this.clusteringPoints.length > 0) {
        this.clusteringPoints.pop();
        this.drawClusteringCanvas();
      }
      return;
    }

    // Left-click: add point
    const [x, y] = AppUtils.pxToData(px, py, canvas);
    this.clusteringPoints.push({ x, y });
    this.drawClusteringCanvas();
  }

  clearClustering() {
    this.clusteringPoints = [];
    this.resetKMeans();
    this.resetHierarchical();
    this.drawClusteringCanvas();
  }

  sampleClustering() {
    this.clusteringPoints = [];

    // Generate 3 clusters with random points
    const centers = [
      { x: 2, y: 2 },
      { x: 8, y: 8 },
      { x: 5, y: 3 }
    ];

    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < 5; i++) {
        const x = centers[c].x + (Math.random() - 0.5) * 3;
        const y = centers[c].y + (Math.random() - 0.5) * 3;
        this.clusteringPoints.push({
          x: Math.max(0, Math.min(10, x)),
          y: Math.max(0, Math.min(10, y))
        });
      }
    }

    this.drawClusteringCanvas();
  }

  drawClusteringCanvas() {
    const assignments = this.kMeansClustering.initialized ? this.kMeansClustering.assignments :
                        (this.hierarchicalClustering.assignments.length > 0 ? this.hierarchicalClustering.assignments : null);
    const centers = this.kMeansClustering.initialized ? this.kMeansClustering.centers : null;
    this.canvas.drawClustering(this.clusteringPoints, assignments, centers);
  }

  initKMeans() {
    try {
      this.kMeansClustering.initialize();
      this.drawClusteringCanvas();
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  nextKMeansIter() {
    try {
      this.kMeansClustering.trainIteration();
      this.drawClusteringCanvas();
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  autoKMeans() {
    try {
      if (!this.kMeansClustering.initialized) {
        this.kMeansClustering.initialize();
      }
      // Override the autoRun to also draw canvas
      const originalAutoRun = this.kMeansClustering.autoRun.bind(this.kMeansClustering);
      const self = this;
      this.kMeansClustering.autoRun = function() {
        if (this.autoRunning) return;
        this.autoRunning = true;

        const maxIterations = 100;
        const runStep = () => {
          if (!this.autoRunning || this.iteration >= maxIterations) {
            this.stopAutoRun();
            return;
          }

          try {
            const converged = this.trainIteration();
            self.drawClusteringCanvas();
            if (converged) {
              this.stopAutoRun();
              return;
            }
            this.autoTimeoutId = setTimeout(runStep, 200);
          } catch (e) {
            this.stopAutoRun();
            window.showToast(e.message, 'error');
          }
        };

        runStep();
      };
      this.kMeansClustering.autoRun();
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  resetKMeans() {
    this.kMeansClustering.stopAutoRun();
    this.kMeansClustering.reset();
    document.getElementById('next-kmeans').disabled = true;
    document.getElementById('kmeans-iter').textContent = '0';
    document.getElementById('kmeans-inertia').textContent = '—';
    document.getElementById('kmeans-equations').style.display = 'none';
    document.getElementById('kmeans-results').style.display = 'none';
    this.drawClusteringCanvas();
  }

  computeHierarchical() {
    try {
      this.hierarchicalClustering.computeClustering();
      this.drawClusteringCanvas();
    } catch (e) {
      window.showToast(e.message, 'error');
    }
  }

  resetHierarchical() {
    this.hierarchicalClustering.reset();
    document.getElementById('hierarchical-equations').style.display = 'none';
    document.getElementById('hierarchical-results').style.display = 'none';

    // Clear dendrogram canvas
    const dendroCanvas = document.getElementById('hierarchical-dendrogram');
    if (dendroCanvas) {
      const ctx = dendroCanvas.getContext('2d');
      ctx.clearRect(0, 0, dendroCanvas.width, dendroCanvas.height);
    }
    this.drawClusteringCanvas();
  }

  initializeFormulas() {
    AppUtils.kRender(document.getElementById('ols-formula'),
      `\\beta=(X^{\\top}X)^{-1}X^{\\top}y,\\quad X=\\begin{bmatrix}x_1&1\\\\\\vdots&\\vdots\\\\x_n&1\\end{bmatrix},\\; \\beta=\\begin{bmatrix}m\\\\b\\end{bmatrix}`, true);
    AppUtils.kRender(document.getElementById('gd-h-formula'), `h_i = m\\,x_i + b`);
    AppUtils.kRender(document.getElementById('gd-j-formula'), `J=\\tfrac{1}{2n}\\sum (h_i-y_i)^2`);
    AppUtils.kRender(document.getElementById('gd-grads-formula'),
      `\\frac{\\partial J}{\\partial m}=\\tfrac{1}{n}\\sum x_i(h_i-y_i),\\; \\frac{\\partial J}{\\partial b}=\\tfrac{1}{n}\\sum (h_i-y_i)`);
    AppUtils.kRender(document.getElementById('gd-update-formula'),
      `m\\leftarrow m-\\eta\\,\\tfrac{\\partial J}{\\partial m},\\; b\\leftarrow b-\\eta\\,\\tfrac{\\partial J}{\\partial b}`);
  }
}

// Initialize app when DOM is loaded and solvers are ready
function bootstrapApp() {
  const depsReady = [
    window.CanvasHandler,
    window.OLSSolver,
    window.GradientDescent,
    window.ManualCalculator,
    window.LogisticRegression,
    window.StatisticsCalculator,
    window.DecisionTree,
    window.RandomForest,
    window.XGBoost,
    window.NeuralNetwork,
    window.RecommendationSystem,
    window.ContentBasedFiltering,
    window.CollaborativeFiltering,
    window.ClusteringManager,
    window.KMeansClustering,
    window.HierarchicalClustering
  ].every(Boolean);

  if (depsReady) {
    window.App = new App();
  } else {
    setTimeout(bootstrapApp, 10);
  }
}

document.addEventListener('DOMContentLoaded', bootstrapApp);
