/* ========================================================
   RAGAVEDA - INTERACTIVE DASHBOARD JAVASCRIPT
   ======================================================== */

// Global App State
const state = {
  selectedSamplePath: null,
  selectedFile: null,
  selectedFilename: null,
  samplesByRaga: {},
  allRagas: {},
  metrics: null,
  isPlaying: false,
  audioContext: null,
  analyser: null,
  audioSourceNode: null,
  animationId: null
};

// DOM Elements
const dom = {
  // Navigation Tabs
  navBtns: document.querySelectorAll('.nav-tab-btn'),
  tabPanels: document.querySelectorAll('.tab-panel'),
  statusLabel: document.getElementById('status-label'),
  systemStatusPill: document.getElementById('system-status-indicator'),

  // Studio Inputs
  ragaPillsSelector: document.getElementById('raga-pills-selector'),
  sampleTracksContainer: document.getElementById('sample-tracks-container'),
  sampleChipsList: document.getElementById('sample-chips-list'),
  uploadDropzone: document.getElementById('upload-dropzone'),
  fileInput: document.getElementById('audio-file-input'),
  currentSelectionBadge: document.getElementById('current-selection-badge'),
  selectedFilenameText: document.getElementById('selected-filename-text'),
  selectedFilesourceText: document.getElementById('selected-filesource-text'),
  btnClearSelection: document.getElementById('btn-clear-selection'),

  // Audio Player
  audioElement: document.getElementById('audio-element'),
  btnPlayPause: document.getElementById('btn-play-pause'),
  progressBarFill: document.getElementById('progress-bar-fill'),
  progressBarBg: document.getElementById('progress-bar-bg'),
  currentTime: document.getElementById('current-time'),
  totalDuration: document.getElementById('total-duration'),
  waveformCanvas: document.getElementById('waveform-canvas'),
  btnRunPrediction: document.getElementById('btn-run-prediction'),

  // Results State Panels
  resultStatusBadge: document.getElementById('result-status-badge'),
  resultsEmptyState: document.getElementById('results-empty-state'),
  resultsLoadingState: document.getElementById('results-loading-state'),
  resultsPopulatedContent: document.getElementById('results-populated-content'),

  // Prediction Details
  predictedRagaName: document.getElementById('predicted-raga-name'),
  predictedThaat: document.getElementById('predicted-thaat'),
  confidenceScoreVal: document.getElementById('confidence-score-val'),
  predictedPrahar: document.getElementById('predicted-prahar'),
  predictedRasa: document.getElementById('predicted-rasa'),
  swaraAaroh: document.getElementById('swara-aaroh'),
  swaraAvroh: document.getElementById('swara-avroh'),
  swaraVadi: document.getElementById('swara-vadi'),
  swaraSamvadi: document.getElementById('swara-samvadi'),
  swaraPakad: document.getElementById('swara-pakad'),
  ragaNarrativeText: document.getElementById('raga-narrative-text'),
  probBarsList: document.getElementById('prob-bars-list'),

  // Spectrogram Tab Buttons & Images
  btnTabMel: document.getElementById('btn-tab-mel'),
  btnTabMfcc: document.getElementById('btn-tab-mfcc'),
  imgMelSpec: document.getElementById('img-mel-spec'),
  imgMfccMap: document.getElementById('img-mfcc-map'),

  // Benchmarks Tab
  statTestAcc: document.getElementById('stat-test-acc'),
  statCnnAcc: document.getElementById('stat-cnn-acc'),
  confusionMatrixContainer: document.getElementById('confusion-matrix-container'),
  reportTableBody: document.getElementById('report-table-body'),

  // Compendium & Dataset Tab
  compendiumCardsContainer: document.getElementById('compendium-cards-container'),
  trainDistList: document.getElementById('train-dist-list'),
  testDistList: document.getElementById('test-dist-list')
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  setupNavigationTabs();
  setupAudioPlayer();
  setupFileUpload();
  setupSpectrogramTabs();
  setupVisualizerCanvas();
  
  // Fetch Initial Data
  fetchSystemStatus();
  fetchSampleFiles();
  fetchMetrics();
  fetchAllRagasCompendium();
});

// 1. Navigation Tabs Switching
function setupNavigationTabs() {
  dom.navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      dom.navBtns.forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      dom.tabPanels.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
      const targetId = btn.getAttribute('data-target');
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add('active');
      }
    });
  });
}

// 2. Fetch System Status
async function fetchSystemStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.status === 'online') {
      const deviceStr = data.cuda_available ? 'CUDA GPU Active' : 'CPU Mode';
      dom.statusLabel.textContent = `Online • ${deviceStr} • Acc: ${data.best_accuracy}%`;
      dom.systemStatusPill.style.color = '#10b981';
      dom.systemStatusPill.style.borderColor = 'rgba(16, 185, 129, 0.4)';
    }
  } catch (err) {
    dom.statusLabel.textContent = 'Server Offline';
    dom.systemStatusPill.style.color = '#f43f5e';
    dom.systemStatusPill.style.borderColor = 'rgba(244, 63, 94, 0.4)';
  }
}

// 3. Fetch Sample Files and Render Pills
async function fetchSampleFiles() {
  try {
    const res = await fetch('/api/samples');
    state.samplesByRaga = await res.json();
    renderRagaPills();
    renderDatasetDistribution();
  } catch (err) {
    console.error('Error fetching sample files:', err);
    dom.ragaPillsSelector.innerHTML = '<span class="error-msg">Could not load samples.</span>';
  }
}

function renderRagaPills() {
  dom.ragaPillsSelector.innerHTML = '';
  const ragas = Object.keys(state.samplesByRaga);

  ragas.forEach((raga, idx) => {
    const btn = document.createElement('button');
    btn.className = `raga-pill-btn ${idx === 0 ? 'active' : ''}`;
    btn.textContent = raga;
    btn.addEventListener('click', () => {
      document.querySelectorAll('.raga-pill-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderSampleChips(raga);
    });
    dom.ragaPillsSelector.appendChild(btn);
  });

  if (ragas.length > 0) {
    renderSampleChips(ragas[0]);
  }
}

function renderSampleChips(raga) {
  const tracks = state.samplesByRaga[raga] || [];
  dom.sampleChipsList.innerHTML = '';

  if (tracks.length === 0) {
    dom.sampleTracksContainer.style.display = 'none';
    return;
  }

  dom.sampleTracksContainer.style.display = 'block';
  tracks.slice(0, 5).forEach((track, idx) => {
    const chip = document.createElement('button');
    chip.className = `sample-chip ${idx === 0 ? 'active' : ''}`;
    chip.textContent = `▶ ${track.name}`;
    chip.title = `Select ${track.name} from ${track.folder}`;
    chip.addEventListener('click', () => {
      document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      selectSample(track.path, track.name, raga);
    });
    dom.sampleChipsList.appendChild(chip);
  });

  // Auto-select first sample of this raga
  if (tracks.length > 0) {
    selectSample(tracks[0].path, tracks[0].name, raga);
  }
}

function selectSample(path, filename, raga) {
  state.selectedSamplePath = path;
  state.selectedFile = null;
  state.selectedFilename = filename;

  // Update selection badge
  dom.selectedFilenameText.textContent = filename;
  dom.selectedFilesourceText.textContent = `Verified ${raga} recording from dataset`;
  dom.currentSelectionBadge.style.display = 'flex';

  // Load into Audio Player
  const audioUrl = `/api/audio-file?path=${encodeURIComponent(path)}`;
  loadAudioIntoPlayer(audioUrl);
  dom.btnRunPrediction.disabled = false;
  resetResultsState();
}

// 4. File Upload (Drag & Drop + Input)
function setupFileUpload() {
  const dropzone = dom.uploadDropzone;
  const input = dom.fileInput;

  dropzone.addEventListener('click', () => input.click());

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, e => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, e => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleUploadedFile(files[0]);
    }
  });

  input.addEventListener('change', e => {
    if (input.files.length > 0) {
      handleUploadedFile(input.files[0]);
    }
  });

  dom.btnClearSelection.addEventListener('click', () => {
    clearSelection();
  });
}

function handleUploadedFile(file) {
  if (!file.name.match(/\.(wav|mp3|ogg|flac)$/i)) {
    alert('Please upload an audio file (.wav or .mp3)');
    return;
  }

  state.selectedFile = file;
  state.selectedSamplePath = null;
  state.selectedFilename = file.name;

  // Clear pill active classes
  document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));

  dom.selectedFilenameText.textContent = file.name;
  dom.selectedFilesourceText.textContent = `User upload (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
  dom.currentSelectionBadge.style.display = 'flex';

  // Create Object URL for player
  const objectUrl = URL.createObjectURL(file);
  loadAudioIntoPlayer(objectUrl);
  dom.btnRunPrediction.disabled = false;
  resetResultsState();
}

function clearSelection() {
  state.selectedFile = null;
  state.selectedSamplePath = null;
  state.selectedFilename = null;
  dom.currentSelectionBadge.style.display = 'none';
  dom.audioElement.pause();
  dom.audioElement.src = '';
  dom.btnPlayPause.disabled = true;
  dom.btnRunPrediction.disabled = true;
  dom.progressBarFill.style.width = '0%';
  dom.currentTime.textContent = '0:00';
  resetResultsState();
}

// 5. Audio Player & Visualizer Engine
function setupAudioPlayer() {
  const audio = dom.audioElement;

  dom.btnPlayPause.addEventListener('click', () => {
    if (audio.paused) {
      audio.play();
    } else {
      audio.pause();
    }
  });

  audio.addEventListener('play', () => {
    state.isPlaying = true;
    dom.btnPlayPause.innerHTML = '<span class="play-icon">⏸</span>';
    initWebAudio();
    drawWaveform();
  });

  audio.addEventListener('pause', () => {
    state.isPlaying = false;
    dom.btnPlayPause.innerHTML = '<span class="play-icon">▶</span>';
  });

  audio.addEventListener('timeupdate', () => {
    if (audio.duration) {
      const pct = (audio.currentTime / audio.duration) * 100;
      dom.progressBarFill.style.width = `${pct}%`;
      dom.currentTime.textContent = formatTime(audio.currentTime);
      dom.totalDuration.textContent = formatTime(audio.duration);
    }
  });

  audio.addEventListener('ended', () => {
    state.isPlaying = false;
    dom.btnPlayPause.innerHTML = '<span class="play-icon">▶</span>';
    dom.progressBarFill.style.width = '0%';
  });

  dom.progressBarBg.addEventListener('click', e => {
    if (audio.duration) {
      const rect = dom.progressBarBg.getBoundingClientRect();
      const clickPos = (e.clientX - rect.left) / rect.width;
      audio.currentTime = clickPos * audio.duration;
    }
  });
}

function loadAudioIntoPlayer(url) {
  const audio = dom.audioElement;
  audio.src = url;
  audio.load();
  dom.btnPlayPause.disabled = false;
  dom.progressBarFill.style.width = '0%';
  dom.currentTime.textContent = '0:00';
}

function formatTime(sec) {
  if (isNaN(sec)) return '0:00';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s < 10 ? '0' : ''}${s}`;
}

// HTML5 Canvas Frequency Equalizer
function setupVisualizerCanvas() {
  const canvas = dom.waveformCanvas;
  const ctx = canvas.getContext('2d');
  drawStaticWaveform(ctx, canvas);
}

function initWebAudio() {
  if (!state.audioContext) {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    state.audioContext = new AudioContext();
    state.analyser = state.audioContext.createAnalyser();
    state.analyser.fftSize = 128;
    state.audioSourceNode = state.audioContext.createMediaElementSource(dom.audioElement);
    state.audioSourceNode.connect(state.analyser);
    state.analyser.connect(state.audioContext.destination);
  }
  if (state.audioContext.state === 'suspended') {
    state.audioContext.resume();
  }
}

function drawWaveform() {
  if (!state.isPlaying || !state.analyser) {
    drawStaticWaveform(dom.waveformCanvas.getContext('2d'), dom.waveformCanvas);
    return;
  }

  const canvas = dom.waveformCanvas;
  const ctx = canvas.getContext('2d');
  const bufferLength = state.analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  state.analyser.getByteFrequencyData(dataArray);

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const barWidth = (canvas.width / bufferLength) * 1.8;
  let barHeight;
  let x = 0;

  for (let i = 0; i < bufferLength; i++) {
    barHeight = (dataArray[i] / 255) * canvas.height * 0.9;

    const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
    gradient.addColorStop(0, '#06b6d4');
    gradient.addColorStop(0.6, '#f59e0b');
    gradient.addColorStop(1, '#fbbf24');

    ctx.fillStyle = gradient;
    ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);

    x += barWidth + 1;
  }

  requestAnimationFrame(drawWaveform);
}

function drawStaticWaveform(ctx, canvas) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
  const bars = 48;
  const barWidth = canvas.width / bars;

  for (let i = 0; i < bars; i++) {
    const height = (Math.sin(i * 0.25) * 0.4 + 0.5) * (canvas.height * 0.6) + 4;
    ctx.fillRect(i * barWidth, (canvas.height - height) / 2, barWidth - 2, height);
  }
}

// 6. Prediction Handler
dom.btnRunPrediction.addEventListener('click', runPrediction);

async function runPrediction() {
  if (!state.selectedFile && !state.selectedSamplePath) {
    alert('Please select or upload an audio file first.');
    return;
  }

  // Show loading state
  dom.resultsEmptyState.style.display = 'none';
  dom.resultsPopulatedContent.style.display = 'none';
  dom.resultsLoadingState.style.display = 'block';
  dom.resultStatusBadge.textContent = 'Analyzing...';
  dom.resultStatusBadge.className = 'badge badge-accent';
  dom.btnRunPrediction.disabled = true;

  const formData = new FormData();
  if (state.selectedFile) {
    formData.append('audio_file', state.selectedFile);
  } else if (state.selectedSamplePath) {
    formData.append('sample_path', state.selectedSamplePath);
  }

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      throw new Error(`Server returned error: ${res.statusText}`);
    }

    const data = await res.json();
    renderPredictionResults(data);

  } catch (err) {
    console.error('Prediction failed:', err);
    alert(`Prediction failed: ${err.message}`);
    dom.resultsLoadingState.style.display = 'none';
    dom.resultsEmptyState.style.display = 'block';
    dom.resultStatusBadge.textContent = 'Error';
    dom.resultStatusBadge.className = 'badge';
  } finally {
    dom.btnRunPrediction.disabled = false;
  }
}

function renderPredictionResults(data) {
  dom.resultsLoadingState.style.display = 'none';
  dom.resultsPopulatedContent.style.display = 'block';
  dom.resultStatusBadge.textContent = 'Analysis Complete';
  dom.resultStatusBadge.className = 'badge badge-accent';

  // 1. Hero banner
  dom.predictedRagaName.textContent = `Raga ${data.predicted_raga}`;
  dom.confidenceScoreVal.textContent = `${data.confidence}%`;

  const info = data.raga_info || {};
  dom.predictedThaat.textContent = `Thaat: ${info.thaat || 'Hindustani'}`;
  dom.predictedPrahar.textContent = info.prahar || 'Classical Time Window';
  dom.predictedRasa.textContent = info.rasa || 'Characteristic Mood';

  // 2. Swara structure
  dom.swaraAaroh.textContent = info.aaroh || 'S R G M P D N S';
  dom.swaraAvroh.textContent = info.avroh || 'S N D P M G R S';
  dom.swaraVadi.textContent = info.vadi || 'Dominant Swara';
  dom.swaraSamvadi.textContent = info.samvadi || 'Sub-dominant Swara';
  dom.swaraPakad.textContent = info.pakad || 'Characteristic Melodic Motif';
  dom.ragaNarrativeText.textContent = info.description || '';

  // 3. Probability bars
  dom.probBarsList.innerHTML = '';
  data.probabilities.forEach(item => {
    const isTop = item.raga === data.predicted_raga;
    const div = document.createElement('div');
    div.className = 'prob-item';
    div.innerHTML = `
      <div class="prob-raga-name" style="${isTop ? 'color: var(--amber-400); font-weight: 700;' : ''}">${item.raga}</div>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" style="width: ${item.probability}%; ${isTop ? 'background: linear-gradient(90deg, #f59e0b, #fbbf24);' : ''}"></div>
      </div>
      <div class="prob-value">${item.probability}%</div>
    `;
    dom.probBarsList.appendChild(div);
  });

  // 4. Spectrogram & MFCC images
  if (data.spectrogram_image) {
    dom.imgMelSpec.src = data.spectrogram_image;
  }
  if (data.mfcc_image) {
    dom.imgMfccMap.src = data.mfcc_image;
  }
}

function resetResultsState() {
  dom.resultsPopulatedContent.style.display = 'none';
  dom.resultsLoadingState.style.display = 'none';
  dom.resultsEmptyState.style.display = 'block';
  dom.resultStatusBadge.textContent = 'Awaiting Audio';
  dom.resultStatusBadge.className = 'badge';
}

// 7. Spectrogram Tab Switching
function setupSpectrogramTabs() {
  dom.btnTabMel.addEventListener('click', () => {
    dom.btnTabMel.classList.add('active');
    dom.btnTabMfcc.classList.remove('active');
    dom.imgMelSpec.classList.add('active');
    dom.imgMfccMap.classList.remove('active');
  });

  dom.btnTabMfcc.addEventListener('click', () => {
    dom.btnTabMfcc.classList.add('active');
    dom.btnTabMel.classList.remove('active');
    dom.imgMfccMap.classList.add('active');
    dom.imgMelSpec.classList.remove('active');
  });
}

// 8. Fetch Metrics and Render Benchmarks Panel
async function fetchMetrics() {
  try {
    const res = await fetch('/api/metrics');
    state.metrics = await res.json();
    renderBenchmarksPanel(state.metrics);
  } catch (err) {
    console.error('Error fetching metrics:', err);
  }
}

function renderBenchmarksPanel(metrics) {
  if (!metrics) return;

  dom.statTestAcc.textContent = `${metrics.ensemble_test_accuracy || 87.23}%`;
  dom.statCnnAcc.textContent = `${metrics.cnn_test_accuracy || 82.98}%`;

  // Render Confusion Matrix
  const cm = metrics.confusion_matrix || [];
  const classes = metrics.classes || [];
  const cmContainer = dom.confusionMatrixContainer;
  cmContainer.innerHTML = '';

  // Header row
  const emptyCorner = document.createElement('div');
  emptyCorner.className = 'cm-cell cm-header';
  emptyCorner.textContent = 'Act\\Pred';
  cmContainer.appendChild(emptyCorner);

  classes.forEach(cls => {
    const th = document.createElement('div');
    th.className = 'cm-cell cm-header';
    th.textContent = cls.slice(0, 3);
    th.title = cls;
    cmContainer.appendChild(th);
  });

  // Data rows
  cm.forEach((row, i) => {
    const rowHeader = document.createElement('div');
    rowHeader.className = 'cm-cell cm-header';
    rowHeader.textContent = classes[i].slice(0, 3);
    rowHeader.title = classes[i];
    cmContainer.appendChild(rowHeader);

    row.forEach((val, j) => {
      const cell = document.createElement('div');
      let valClass = 'cm-val-0';
      if (val > 0) {
        if (i === j) {
          valClass = val >= 8 ? 'cm-val-high' : 'cm-val-mid';
        } else {
          valClass = 'cm-val-low';
        }
      }
      cell.className = `cm-cell ${valClass}`;
      cell.textContent = val;
      cell.title = `Actual: ${classes[i]} | Predicted: ${classes[j]} (${val} samples)`;
      cmContainer.appendChild(cell);
    });
  });

  // Render Classification Report Table
  const report = metrics.classification_report || {};
  dom.reportTableBody.innerHTML = '';

  classes.forEach(cls => {
    const rowData = report[cls] || {};
    const prec = rowData.precision !== undefined ? (rowData.precision * 100).toFixed(1) : '-';
    const rec = rowData.recall !== undefined ? (rowData.recall * 100).toFixed(1) : '-';
    const f1 = rowData['f1-score'] !== undefined ? (rowData['f1-score'] * 100).toFixed(1) : '-';
    const supp = rowData.support !== undefined ? rowData.support : '-';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="font-weight: 600; color: #ffffff;">${cls}</td>
      <td class="${prec === '100.0' ? 'score-perfect' : ''}">${prec}%</td>
      <td class="${rec === '100.0' ? 'score-perfect' : ''}">${rec}%</td>
      <td>${f1}%</td>
      <td>${supp}</td>
    `;
    dom.reportTableBody.appendChild(tr);
  });
}

// 9. Fetch Compendium & Render Compendium Cards
async function fetchAllRagasCompendium() {
  try {
    const res = await fetch('/api/all-ragas');
    state.allRagas = await res.json();
    renderCompendiumCards(state.allRagas);
  } catch (err) {
    console.error('Error fetching compendium:', err);
  }
}

function renderCompendiumCards(ragasDict) {
  const container = dom.compendiumCardsContainer;
  container.innerHTML = '';

  Object.keys(ragasDict).forEach(ragaName => {
    const info = ragasDict[ragaName];
    const card = document.createElement('div');
    card.className = 'raga-card';
    card.innerHTML = `
      <div class="raga-card-header">
        <div class="raga-card-title">${ragaName}</div>
        <div class="raga-card-thaat">${info.thaat} Thaat</div>
      </div>
      <div class="raga-card-info-row">
        <span><strong>Time:</strong> ${info.prahar}</span>
      </div>
      <div class="raga-card-info-row">
        <span><strong>Mood:</strong> ${info.rasa}</span>
      </div>
      <div class="raga-card-notes">
        <div><strong>Aaroh:</strong> ${info.aaroh}</div>
        <div><strong>Avroh:</strong> ${info.avroh}</div>
      </div>
      <div class="raga-card-info-row">
        <span><strong>Vadi:</strong> ${info.vadi}</span>
        <span><strong>Samvadi:</strong> ${info.samvadi}</span>
      </div>
      <p class="raga-card-desc">${info.description}</p>
    `;
    container.appendChild(card);
  });
}

// 10. Render Dataset Distribution
function renderDatasetDistribution() {
  const trainCounts = {
    "Asavari": 10, "Bageshree": 12, "Bhairava": 25, "Bhairavi": 10,
    "Bhoopali": 40, "Darbari": 10, "Malkauns": 10, "Sarang": 10, "Yaman": 42
  };
  const testCounts = {
    "Asavari": 10, "Bageshree": 12, "Bhairava": 12, "Bhairavi": 10,
    "Bhoopali": 10, "Darbari": 10, "Malkauns": 10, "Sarang": 10, "Yaman": 10
  };

  dom.trainDistList.innerHTML = '';
  Object.keys(trainCounts).forEach(raga => {
    const item = document.createElement('div');
    item.className = 'dataset-dist-item';
    item.innerHTML = `<span>Raga ${raga}</span><span class="dist-count">${trainCounts[raga]} tracks</span>`;
    dom.trainDistList.appendChild(item);
  });

  dom.testDistList.innerHTML = '';
  Object.keys(testCounts).forEach(raga => {
    const item = document.createElement('div');
    item.className = 'dataset-dist-item';
    item.innerHTML = `<span>Raga ${raga}</span><span class="dist-count">${testCounts[raga]} tracks</span>`;
    dom.testDistList.appendChild(item);
  });
}
