// ============================================================================
// CLUSTERING
// ============================================================================

class ClusteringManager {
  constructor() {
    this.reset();
  }

  reset() {
    this.dataPoints = [];
    this.pointNames = [];
  }

  getDataFromApp() {
    // Get points from the app's clusteringPoints array
    const points = window.App ? window.App.clusteringPoints : [];
    const pointNames = points.map((_, i) => `P${i + 1}`);

    this.dataPoints = points;
    this.pointNames = pointNames;
    return { points, pointNames };
  }
}

class KMeansClustering {
  constructor(clusteringManager) {
    this.manager = clusteringManager;
    this.reset();
  }

  reset() {
    this.initialized = false;
    this.iteration = 0;
    this.centers = [];
    this.assignments = [];
    this.inertiaHistory = [];
    this.autoRunning = false;
    this.autoTimeoutId = null;
  }

  initialize() {
    const { points, pointNames } = this.manager.getDataFromApp();
    if (points.length < 2) {
      window.showToast('Add at least 2 points to the canvas first.', 'warn');
      return;
    }

    const kInput = document.getElementById('kmeans-k');
    let k = parseInt(kInput.value);
    if (isNaN(k) || k < 1) k = 1;
    if (k > points.length) {
      window.showToast(`K (${k}) exceeds point count (${points.length}); using K=${points.length}.`, 'warn');
      k = points.length;
      kInput.value = k;
    }

    // Initialize K random centers using k-means++ style
    this.centers = [];
    const usedIndices = new Set();

    // First center is random
    let idx = Math.floor(Math.random() * points.length);
    usedIndices.add(idx);
    this.centers.push({ x: points[idx].x, y: points[idx].y });

    // Remaining centers: pick points with probability proportional to distance squared
    for (let i = 1; i < k; i++) {
      const distances = points.map((p, pIdx) => {
        if (usedIndices.has(pIdx)) return 0;
        let minDist = Infinity;
        this.centers.forEach(c => {
          const d = Math.sqrt(Math.pow(p.x - c.x, 2) + Math.pow(p.y - c.y, 2));
          if (d < minDist) minDist = d;
        });
        return minDist * minDist;
      });

      const totalDist = distances.reduce((s, d) => s + d, 0);
      let r = Math.random() * totalDist;
      let chosenIdx = 0;
      for (let j = 0; j < distances.length; j++) {
        r -= distances[j];
        if (r <= 0) {
          chosenIdx = j;
          break;
        }
      }

      usedIndices.add(chosenIdx);
      this.centers.push({ x: points[chosenIdx].x, y: points[chosenIdx].y });
    }

    this.initialized = true;
    this.iteration = 0;
    this.inertiaHistory = [];
    this.assignments = new Array(points.length).fill(0);

    document.getElementById('next-kmeans').disabled = false;

    // Display algorithm equations
    this.displayEquations(k);

    // Show results sections
    document.getElementById('kmeans-equations').style.display = 'block';
    document.getElementById('kmeans-results').style.display = 'block';

    // Initial assignment
    this.assignPoints();
    this.displayResults(points, pointNames);
  }

  displayEquations(k) {
    const initEq = `\\text{Randomly select } K = ${k} \\text{ initial centroids from data points}\\\\
\\text{Using k-means++ initialization for better starting positions}`;
    AppUtils.kRender(document.getElementById('kmeans-eq-init'), initEq, true);

    const assignEq = `\\text{For each point } \\mathbf{x}_i, \\text{ assign to nearest centroid:}\\\\
c_i = \\arg\\min_j \\|\\mathbf{x}_i - \\boldsymbol{\\mu}_j\\|^2\\\\
\\text{where } \\|\\mathbf{a} - \\mathbf{b}\\| = \\sqrt{(a_x - b_x)^2 + (a_y - b_y)^2}`;
    AppUtils.kRender(document.getElementById('kmeans-eq-assign'), assignEq, true);

    const updateEq = `\\text{Update each centroid as the mean of assigned points:}\\\\
\\boldsymbol{\\mu}_j = \\frac{1}{|C_j|} \\sum_{\\mathbf{x}_i \\in C_j} \\mathbf{x}_i`;
    AppUtils.kRender(document.getElementById('kmeans-eq-update'), updateEq, true);

    const lossEq = `\\text{Inertia (within-cluster sum of squares):}\\\\
\\text{WCSS} = \\sum_{j=1}^{K} \\sum_{\\mathbf{x}_i \\in C_j} \\|\\mathbf{x}_i - \\boldsymbol{\\mu}_j\\|^2`;
    AppUtils.kRender(document.getElementById('kmeans-eq-loss'), lossEq, true);
  }

  assignPoints() {
    const { points } = this.manager.getDataFromApp();

    // Assign each point to nearest center
    this.assignments = points.map(point => {
      let minDist = Infinity;
      let nearestCluster = 0;

      this.centers.forEach((center, clusterIdx) => {
        const dist = Math.sqrt(
          Math.pow(point.x - center.x, 2) + Math.pow(point.y - center.y, 2)
        );
        if (dist < minDist) {
          minDist = dist;
          nearestCluster = clusterIdx;
        }
      });

      return nearestCluster;
    });
  }

  trainIteration() {
    if (!this.initialized) {
      window.showToast('Click "Initialize" before iterating.', 'warn');
      return;
    }

    const { points, pointNames } = this.manager.getDataFromApp();
    const k = this.centers.length;

    // Store old centers to check convergence
    const oldCenters = this.centers.map(c => ({ x: c.x, y: c.y }));

    // STEP 1: Assignment - assign points to nearest center
    this.assignPoints();

    // STEP 2: Update centers
    const newCenters = [];
    for (let c = 0; c < k; c++) {
      const clusterPoints = points.filter((_, idx) => this.assignments[idx] === c);

      if (clusterPoints.length > 0) {
        const sumX = clusterPoints.reduce((s, p) => s + p.x, 0);
        const sumY = clusterPoints.reduce((s, p) => s + p.y, 0);
        newCenters.push({
          x: sumX / clusterPoints.length,
          y: sumY / clusterPoints.length
        });
      } else {
        // Keep old center if cluster is empty
        newCenters.push({ x: this.centers[c].x, y: this.centers[c].y });
      }
    }
    this.centers = newCenters;

    // STEP 3: Compute inertia
    let inertia = 0;
    points.forEach((point, idx) => {
      const center = this.centers[this.assignments[idx]];
      const dist = Math.sqrt(
        Math.pow(point.x - center.x, 2) + Math.pow(point.y - center.y, 2)
      );
      inertia += dist * dist;
    });
    this.inertiaHistory.push(inertia);

    // Update iteration counter
    this.iteration++;

    // Update UI
    document.getElementById('kmeans-iter').textContent = this.iteration;
    document.getElementById('kmeans-inertia').textContent = inertia.toFixed(4);

    // Display results
    this.displayResults(points, pointNames);

    // Check for convergence
    let converged = true;
    for (let c = 0; c < k; c++) {
      const dx = Math.abs(this.centers[c].x - oldCenters[c].x);
      const dy = Math.abs(this.centers[c].y - oldCenters[c].y);
      if (dx > 0.0001 || dy > 0.0001) {
        converged = false;
        break;
      }
    }

    return converged;
  }

  displayResults(points, pointNames) {
    const k = this.centers.length;

    // Display step details
    let stepText = `<strong>Iteration ${this.iteration}:</strong><br>`;

    // Count points per cluster
    const clusterCounts = new Array(k).fill(0);
    this.assignments.forEach(a => clusterCounts[a]++);

    stepText += `Cluster sizes: ${clusterCounts.map((c, i) => `C${i+1}: ${c}`).join(', ')}<br>`;

    if (this.inertiaHistory.length > 0) {
      const currentInertia = this.inertiaHistory[this.inertiaHistory.length - 1];
      stepText += `Current inertia: ${currentInertia.toFixed(4)}`;

      if (this.inertiaHistory.length > 1) {
        const improvement = this.inertiaHistory[this.inertiaHistory.length - 2] - currentInertia;
        stepText += ` (change: ${improvement >= 0 ? '-' : '+'}${Math.abs(improvement).toFixed(4)})`;
      }
    }

    document.getElementById('kmeans-step-display').innerHTML = stepText;

    // Display assignments table
    this.displayAssignments(points, pointNames);

    // Display centers table
    this.displayCenters();

    // Update metrics
    this.updateMetrics(points);
  }

  displayAssignments(points, pointNames) {
    const table = document.getElementById('kmeans-assignments-table');
    table.innerHTML = '';

    // Header
    const headerRow = document.createElement('tr');
    ['Point', 'X', 'Y', 'Cluster', 'Distance'].forEach(text => {
      const th = document.createElement('th');
      th.textContent = text;
      headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    // Colors for clusters
    const colors = ['#fca5a5', '#86efac', '#93c5fd', '#fcd34d', '#d8b4fe', '#fda4af', '#a5f3fc', '#fdba74', '#c4b5fd', '#bef264'];

    // Body
    points.forEach((point, idx) => {
      const tr = document.createElement('tr');
      const clusterIdx = this.assignments[idx];
      const center = this.centers[clusterIdx];
      const dist = Math.sqrt(Math.pow(point.x - center.x, 2) + Math.pow(point.y - center.y, 2));

      // Point name
      let td = document.createElement('td');
      td.textContent = pointNames[idx];
      tr.appendChild(td);

      // X
      td = document.createElement('td');
      td.textContent = point.x.toFixed(2);
      tr.appendChild(td);

      // Y
      td = document.createElement('td');
      td.textContent = point.y.toFixed(2);
      tr.appendChild(td);

      // Cluster
      td = document.createElement('td');
      td.textContent = `C${clusterIdx + 1}`;
      td.style.fontWeight = 'bold';
      td.style.backgroundColor = colors[clusterIdx % colors.length];
      tr.appendChild(td);

      // Distance
      td = document.createElement('td');
      td.textContent = dist.toFixed(4);
      tr.appendChild(td);

      table.appendChild(tr);
    });
  }

  displayCenters() {
    const table = document.getElementById('kmeans-centers-table');
    table.innerHTML = '';

    // Header
    const headerRow = document.createElement('tr');
    ['Centroid', 'X', 'Y', 'Points'].forEach(text => {
      const th = document.createElement('th');
      th.textContent = text;
      headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    // Colors for clusters
    const colors = ['#fca5a5', '#86efac', '#93c5fd', '#fcd34d', '#d8b4fe', '#fda4af', '#a5f3fc', '#fdba74', '#c4b5fd', '#bef264'];

    // Count points per cluster
    const clusterCounts = new Array(this.centers.length).fill(0);
    this.assignments.forEach(a => clusterCounts[a]++);

    // Body
    this.centers.forEach((center, idx) => {
      const tr = document.createElement('tr');

      // Centroid name
      let td = document.createElement('td');
      td.textContent = `C${idx + 1}`;
      td.style.fontWeight = 'bold';
      td.style.backgroundColor = colors[idx % colors.length];
      tr.appendChild(td);

      // X
      td = document.createElement('td');
      td.textContent = center.x.toFixed(4);
      tr.appendChild(td);

      // Y
      td = document.createElement('td');
      td.textContent = center.y.toFixed(4);
      tr.appendChild(td);

      // Points count
      td = document.createElement('td');
      td.textContent = clusterCounts[idx];
      tr.appendChild(td);

      table.appendChild(tr);
    });
  }

  updateMetrics(points) {
    // Inertia
    const inertia = this.inertiaHistory.length > 0 ?
      this.inertiaHistory[this.inertiaHistory.length - 1] : 0;
    document.getElementById('kmeans-inertia-val').textContent = inertia.toFixed(4);
    document.getElementById('kmeans-n-points').textContent = points.length;

    // Silhouette score (simplified version)
    if (this.centers.length > 1 && points.length > this.centers.length) {
      let silhouetteSum = 0;

      points.forEach((point, idx) => {
        const cluster = this.assignments[idx];

        // a: average distance to points in same cluster
        let aSum = 0;
        let aCount = 0;
        points.forEach((other, otherIdx) => {
          if (idx !== otherIdx && this.assignments[otherIdx] === cluster) {
            aSum += Math.sqrt(Math.pow(point.x - other.x, 2) + Math.pow(point.y - other.y, 2));
            aCount++;
          }
        });
        const a = aCount > 0 ? aSum / aCount : 0;

        // b: minimum average distance to points in other clusters
        let b = Infinity;
        for (let c = 0; c < this.centers.length; c++) {
          if (c !== cluster) {
            let bSum = 0;
            let bCount = 0;
            points.forEach((other, otherIdx) => {
              if (this.assignments[otherIdx] === c) {
                bSum += Math.sqrt(Math.pow(point.x - other.x, 2) + Math.pow(point.y - other.y, 2));
                bCount++;
              }
            });
            if (bCount > 0) {
              const avgDist = bSum / bCount;
              if (avgDist < b) b = avgDist;
            }
          }
        }

        if (b === Infinity) b = 0;
        const s = Math.max(a, b) > 0 ? (b - a) / Math.max(a, b) : 0;
        silhouetteSum += s;
      });

      const silhouette = silhouetteSum / points.length;
      document.getElementById('kmeans-silhouette').textContent = silhouette.toFixed(4);
    } else {
      document.getElementById('kmeans-silhouette').textContent = '—';
    }
  }

  autoRun() {
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
  }

  stopAutoRun() {
    this.autoRunning = false;
    if (this.autoTimeoutId) {
      clearTimeout(this.autoTimeoutId);
      this.autoTimeoutId = null;
    }
  }
}

class HierarchicalClustering {
  constructor(clusteringManager) {
    this.manager = clusteringManager;
    this.reset();
  }

  reset() {
    this.mergeHistory = [];
    this.assignments = [];
    this.dendrogram = [];
  }

  computeClustering() {
    const { points, pointNames } = this.manager.getDataFromApp();
    if (points.length < 2) {
      window.showToast('Add at least 2 points to the canvas first.', 'warn');
      return;
    }

    const linkage = document.getElementById('hierarchical-linkage').value;
    const nClustersInput = document.getElementById('hierarchical-n-clusters');
    let nClusters = parseInt(nClustersInput.value);
    if (isNaN(nClusters) || nClusters < 1) nClusters = 1;

    if (nClusters > points.length) {
      window.showToast(`Cluster count (${nClusters}) exceeds point count (${points.length}); using ${points.length}.`, 'warn');
      nClusters = points.length;
      nClustersInput.value = nClusters;
    }

    const n = points.length;

    // Initialize: each point is its own cluster
    let clusters = points.map((p, i) => ({
      id: i,
      points: [i],
      x: p.x,
      y: p.y
    }));

    // Distance matrix
    const distMatrix = [];
    for (let i = 0; i < n; i++) {
      distMatrix[i] = [];
      for (let j = 0; j < n; j++) {
        if (i === j) {
          distMatrix[i][j] = Infinity;
        } else {
          distMatrix[i][j] = Math.sqrt(
            Math.pow(points[i].x - points[j].x, 2) +
            Math.pow(points[i].y - points[j].y, 2)
          );
        }
      }
    }

    // Display algorithm equations
    this.displayEquations(linkage);

    // Merge history for display
    this.mergeHistory = [];
    this.dendrogram = [];

    let clusterIdCounter = n;
    const clusterMap = new Map();
    clusters.forEach(c => clusterMap.set(c.id, c));

    // Agglomerative clustering
    while (clusters.length > 1) {
      // Find closest pair of clusters
      let minDist = Infinity;
      let mergeI = -1;
      let mergeJ = -1;

      for (let i = 0; i < clusters.length; i++) {
        for (let j = i + 1; j < clusters.length; j++) {
          const dist = this.clusterDistance(clusters[i], clusters[j], points, linkage);
          if (dist < minDist) {
            minDist = dist;
            mergeI = i;
            mergeJ = j;
          }
        }
      }

      // Merge clusters
      const clusterA = clusters[mergeI];
      const clusterB = clusters[mergeJ];

      const mergedPoints = [...clusterA.points, ...clusterB.points];
      const mergedX = mergedPoints.reduce((s, idx) => s + points[idx].x, 0) / mergedPoints.length;
      const mergedY = mergedPoints.reduce((s, idx) => s + points[idx].y, 0) / mergedPoints.length;

      const newCluster = {
        id: clusterIdCounter++,
        points: mergedPoints,
        x: mergedX,
        y: mergedY,
        left: clusterA.id,
        right: clusterB.id,
        distance: minDist
      };

      clusterMap.set(newCluster.id, newCluster);

      // Record merge
      this.mergeHistory.push({
        cluster1: clusterA.points.map(i => pointNames[i]).join(', '),
        cluster2: clusterB.points.map(i => pointNames[i]).join(', '),
        distance: minDist,
        newCluster: newCluster.id
      });

      this.dendrogram.push({
        left: clusterA.id,
        right: clusterB.id,
        distance: minDist,
        id: newCluster.id,
        size: mergedPoints.length
      });

      // Remove merged clusters and add new one
      clusters = clusters.filter((_, idx) => idx !== mergeI && idx !== mergeJ);
      clusters.push(newCluster);
    }

    // Cut dendrogram to get nClusters
    this.assignments = this.cutDendrogram(points.length, nClusters, clusterMap);

    // Display results
    this.displayResults(points, pointNames, nClusters);

    // Draw dendrogram
    this.drawDendrogram(points.length, pointNames, clusterMap);

    // Show results
    document.getElementById('hierarchical-equations').style.display = 'block';
    document.getElementById('hierarchical-results').style.display = 'block';
  }

  clusterDistance(clusterA, clusterB, points, linkage) {
    const distances = [];

    clusterA.points.forEach(i => {
      clusterB.points.forEach(j => {
        const dist = Math.sqrt(
          Math.pow(points[i].x - points[j].x, 2) +
          Math.pow(points[i].y - points[j].y, 2)
        );
        distances.push(dist);
      });
    });

    switch (linkage) {
      case 'single':
        return Math.min(...distances);
      case 'complete':
        return Math.max(...distances);
      case 'average':
      default:
        return distances.reduce((s, d) => s + d, 0) / distances.length;
    }
  }

  cutDendrogram(n, nClusters, clusterMap) {
    // Start with all points in separate clusters
    const assignments = new Array(n).fill(-1);

    if (nClusters >= n) {
      // Each point is its own cluster
      for (let i = 0; i < n; i++) {
        assignments[i] = i;
      }
      return assignments;
    }

    // Find the merge step where we go from nClusters to nClusters-1
    const numMerges = n - nClusters;

    // Get the clusters after numMerges merges
    const finalClusters = [];
    const merged = new Set();

    for (let i = 0; i < numMerges; i++) {
      const merge = this.dendrogram[i];
      merged.add(merge.left);
      merged.add(merge.right);
    }

    // Collect final cluster IDs
    // Original points that were never merged
    for (let i = 0; i < n; i++) {
      if (!merged.has(i)) {
        finalClusters.push(i);
      }
    }

    // Plus the merged clusters at the cut point
    for (let i = 0; i < numMerges; i++) {
      const merge = this.dendrogram[i];
      if (!merged.has(merge.id)) {
        finalClusters.push(merge.id);
      }
    }

    // Assign points to clusters
    finalClusters.forEach((clusterId, clusterIdx) => {
      const cluster = clusterMap.get(clusterId);
      if (cluster) {
        cluster.points.forEach(pointIdx => {
          assignments[pointIdx] = clusterIdx;
        });
      }
    });

    return assignments;
  }

  displayEquations(linkage) {
    const initEq = `\\text{Compute pairwise distance matrix:}\\\\
D_{ij} = \\|\\mathbf{x}_i - \\mathbf{x}_j\\| = \\sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}`;
    AppUtils.kRender(document.getElementById('hierarchical-eq-init'), initEq, true);

    const findEq = `\\text{Find pair of clusters with minimum distance:}\\\\
(C_a, C_b) = \\arg\\min_{i,j} d(C_i, C_j)`;
    AppUtils.kRender(document.getElementById('hierarchical-eq-find'), findEq, true);

    let linkageFormula;
    switch (linkage) {
      case 'single':
        linkageFormula = `d(C_a, C_b) = \\min_{\\mathbf{x} \\in C_a, \\mathbf{y} \\in C_b} \\|\\mathbf{x} - \\mathbf{y}\\|`;
        break;
      case 'complete':
        linkageFormula = `d(C_a, C_b) = \\max_{\\mathbf{x} \\in C_a, \\mathbf{y} \\in C_b} \\|\\mathbf{x} - \\mathbf{y}\\|`;
        break;
      case 'average':
      default:
        linkageFormula = `d(C_a, C_b) = \\frac{1}{|C_a||C_b|} \\sum_{\\mathbf{x} \\in C_a} \\sum_{\\mathbf{y} \\in C_b} \\|\\mathbf{x} - \\mathbf{y}\\|`;
    }

    const mergeEq = `\\text{Merge clusters } C_a \\text{ and } C_b \\text{ into } C_{new}\\\\
\\text{Linkage: ${linkage.charAt(0).toUpperCase() + linkage.slice(1)}}\\\\
${linkageFormula}`;
    AppUtils.kRender(document.getElementById('hierarchical-eq-merge'), mergeEq, true);

    const updateEq = `\\text{Update distance matrix:}\\\\
\\text{Remove rows/columns for } C_a, C_b\\\\
\\text{Add row/column for } C_{new}\\\\
\\text{Repeat until one cluster remains}`;
    AppUtils.kRender(document.getElementById('hierarchical-eq-update'), updateEq, true);
  }

  displayResults(points, pointNames, nClusters) {
    // Display merge history
    let historyText = '<strong>Merge sequence:</strong><br>';
    this.mergeHistory.forEach((merge, idx) => {
      historyText += `${idx + 1}. Merge {${merge.cluster1}} and {${merge.cluster2}} (distance: ${merge.distance.toFixed(4)})<br>`;
    });
    document.getElementById('hierarchical-merge-history').innerHTML = historyText;

    // Display assignments table
    const table = document.getElementById('hierarchical-assignments-table');
    table.innerHTML = '';

    // Header
    const headerRow = document.createElement('tr');
    ['Point', 'X', 'Y', 'Cluster'].forEach(text => {
      const th = document.createElement('th');
      th.textContent = text;
      headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    // Colors for clusters
    const colors = ['#fca5a5', '#86efac', '#93c5fd', '#fcd34d', '#d8b4fe', '#fda4af', '#a5f3fc', '#fdba74', '#c4b5fd', '#bef264'];

    // Body
    points.forEach((point, idx) => {
      const tr = document.createElement('tr');
      const clusterIdx = this.assignments[idx];

      // Point name
      let td = document.createElement('td');
      td.textContent = pointNames[idx];
      tr.appendChild(td);

      // X
      td = document.createElement('td');
      td.textContent = point.x.toFixed(2);
      tr.appendChild(td);

      // Y
      td = document.createElement('td');
      td.textContent = point.y.toFixed(2);
      tr.appendChild(td);

      // Cluster
      td = document.createElement('td');
      td.textContent = `C${clusterIdx + 1}`;
      td.style.fontWeight = 'bold';
      td.style.backgroundColor = colors[clusterIdx % colors.length];
      tr.appendChild(td);

      table.appendChild(tr);
    });

    // Update metrics
    document.getElementById('hierarchical-n-clusters-val').textContent = nClusters;
    document.getElementById('hierarchical-n-points').textContent = points.length;
  }

  drawDendrogram(n, pointNames, clusterMap) {
    const canvas = document.getElementById('hierarchical-dendrogram');
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    if (this.dendrogram.length === 0) return;

    const marginL = 40;
    const marginR = 20;
    const marginT = 20;
    const marginB = 40;

    const plotW = w - marginL - marginR;
    const plotH = h - marginT - marginB;

    // Find max distance for scaling
    const maxDist = Math.max(...this.dendrogram.map(d => d.distance));

    // Calculate x positions for each leaf (original point)
    const leafPositions = {};
    const leafSpacing = plotW / (n + 1);
    for (let i = 0; i < n; i++) {
      leafPositions[i] = marginL + (i + 1) * leafSpacing;
    }

    // Calculate positions for merged clusters
    const clusterPositions = { ...leafPositions };

    this.dendrogram.forEach(merge => {
      const leftPos = clusterPositions[merge.left];
      const rightPos = clusterPositions[merge.right];
      clusterPositions[merge.id] = (leftPos + rightPos) / 2;
    });

    // Draw the dendrogram
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;

    this.dendrogram.forEach(merge => {
      const leftX = clusterPositions[merge.left];
      const rightX = clusterPositions[merge.right];
      const mergeX = clusterPositions[merge.id];
      const mergeY = marginT + plotH - (merge.distance / maxDist) * plotH;

      // Get heights of child clusters
      const leftCluster = clusterMap.get(merge.left);
      const rightCluster = clusterMap.get(merge.right);

      const leftY = leftCluster && leftCluster.distance !== undefined ?
        marginT + plotH - (leftCluster.distance / maxDist) * plotH :
        marginT + plotH;
      const rightY = rightCluster && rightCluster.distance !== undefined ?
        marginT + plotH - (rightCluster.distance / maxDist) * plotH :
        marginT + plotH;

      // Draw vertical lines from children to merge height
      ctx.beginPath();
      ctx.moveTo(leftX, leftY);
      ctx.lineTo(leftX, mergeY);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(rightX, rightY);
      ctx.lineTo(rightX, mergeY);
      ctx.stroke();

      // Draw horizontal line connecting children
      ctx.beginPath();
      ctx.moveTo(leftX, mergeY);
      ctx.lineTo(rightX, mergeY);
      ctx.stroke();
    });

    // Draw leaf labels
    ctx.fillStyle = '#374151';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';

    for (let i = 0; i < n; i++) {
      ctx.fillText(pointNames[i], leafPositions[i], h - marginB + 15);
    }

    // Draw y-axis (distance)
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(marginL - 5, marginT);
    ctx.lineTo(marginL - 5, marginT + plotH);
    ctx.stroke();

    // Y-axis labels
    ctx.textAlign = 'right';
    const numTicks = 5;
    for (let i = 0; i <= numTicks; i++) {
      const y = marginT + plotH - (i / numTicks) * plotH;
      const val = (i / numTicks) * maxDist;
      ctx.fillText(val.toFixed(1), marginL - 10, y + 4);
    }

    // Y-axis title
    ctx.save();
    ctx.translate(10, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Distance', 0, 0);
    ctx.restore();
  }
}

// Export classes to global scope
window.ClusteringManager = ClusteringManager;
window.KMeansClustering = KMeansClustering;
window.HierarchicalClustering = HierarchicalClustering;
