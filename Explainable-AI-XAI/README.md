# Machine Learning & XAI Platform

An interactive web application for learning machine learning algorithms through visualization. Click to add data points, watch algorithms train step-by-step, and inspect how each model arrives at its predictions.

![Application Screenshot](screenshots/app-preview.png)

## Features

### Supervised learning
- **Linear models**: Ordinary Least Squares, Gradient Descent (with iteration-by-iteration loss visualization), Manual statistics (TSS/ESS/RSS), Logistic Regression with confusion matrix and Youden's J threshold.
- **Nonlinear models**: Decision Tree with step-by-step recursive splitting, Random Forest with bootstrap sampling and out-of-bag validation, XGBoost with sequential boosting and loss history, Feedforward Neural Network with configurable architecture (ReLU/Tanh/Sigmoid, L2 regularization, dropout), Vanilla RNN (Elman) and LSTM that treat canvas points as a 1-D time series and visualise the recurrence (hidden state, cell state, gate activations) one time step at a time.
- Each supports both **regression** and **classification** tasks.

### Statistics
- Pearson correlation, two-sample t-test (with directional alternatives), one-way ANOVA — each with distribution plots and full derivations.

### Recommendation Systems
- **Content-based filtering** with cosine and Euclidean similarity.
- **Collaborative filtering** with matrix factorization trained by gradient descent.
- Interactive rating and feature matrices.

### Clustering
- **K-Means** with k-means++ initialization, iteration-by-iteration centroid movement, inertia history.
- **Agglomerative hierarchical** clustering with single/complete/average linkage and dendrogram.

### LLMs
- **Theoretical transformer** &mdash; type a sentence, then step through tokenization, embedding lookup, sinusoidal positional encoding, Q/K/V projections, scaled-dot-product attention scores, row-wise softmax, and the final attended output. Every matrix is laid out as a table so the reader can follow the arithmetic; weights are seeded for reproducibility.
- **RAG pipeline** &mdash; edit a small corpus and a query. The tab tokenizes, builds the vocabulary and IDF, computes the TF&ndash;IDF matrix, cosine-similarity-ranks documents, returns top-k, and assembles the augmented prompt that would be sent to a generator.

## Honest evaluation: train/test split

A header-level **Train/test split** control holds out a fraction of points as a test set. Held-out points render as hollow orange rings on the canvas (rather than solid dots). Models are fit on the training subset only, so reported metrics reflect generalization rather than training-set fit. Use `Shift+click` to explicitly mark a point as test rather than letting the random split decide; click `Reshuffle` to reroll which points are held out.

## UI/UX

- Responsive canvases that scale with the viewport (no more 800×500 fixed boxes).
- Light/dark theme toggle in the header, persisted in `localStorage`. Auto-detects the system preference on first load.
- Touch support for adding/dragging points on mobile.
- Keyboard navigation for top-level tabs (Arrow keys, Enter/Space).
- Inline toast notifications for invalid inputs (replaces native `alert()` popups).
- Input bounds enforced for K-Means K, hierarchical cluster count, NN learning rate, hidden-layer string, and epoch cap.

## Setup

### Local

```bash
pip install -r requirements.txt
python src/app.py
```

Visit http://127.0.0.1:5000.

Configurable via environment variables:

```bash
XAI_HOST=0.0.0.0 XAI_PORT=8080 FLASK_ENV=development python src/app.py
```

### Docker

```bash
docker build -t xai .
docker run -p 5000:5000 xai
```

The image runs under gunicorn on port 5000 by default.

## Usage

- **Click** to add a point, **drag** to move, **right-click** to delete.
- **Shift+click** adds a test point when the train/test split is enabled.
- Use the per-tab controls to fit, step, auto-run, or reset.
- Toggle **Residuals** in the header to overlay residual lines on regression plots.

## Architecture

- **Backend**: Flask 3, single-route static app (`src/app.py`). Production-served via gunicorn (`Dockerfile`).
- **Frontend**: Vanilla JavaScript, no build step.
  - `src/static/utils.js` — canvas handling, ResizeObserver-driven responsive rendering, KaTeX formula queue, toast helper.
  - `src/static/app.js` — application controller, train/test split state, event wiring.
  - `src/static/solvers/{linear-models,nonlinear-models,statistics,recsys,clustering}.js` — algorithm implementations.
  - `src/static/styles.css` — CSS variables, light/dark themes, responsive layout.
- KaTeX via CDN for formula rendering.

## Roadmap

- True Explainable-AI techniques wired per model: decision-path highlighting (DT), per-tree vote breakdown and permutation importance (RF), exact tree-additive waterfall attribution (XGBoost), saliency heatmaps and per-feature gradient attribution (NN), per-feature contribution tables (recommender).
- 2D-feature classification mode so decision boundaries can be non-vertical.
