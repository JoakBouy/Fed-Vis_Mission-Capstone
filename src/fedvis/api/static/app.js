/**
 * Fed-Vis Frontend
 *
 * Handles file upload, API calls, and slice visualization.
 */

const API = '';  // same origin
let selectedFile = null;
let inputVolume = null;  // numpy array from uploaded file
let maskVolume = null;   // numpy array from prediction
let maskBlob = null;     // raw blob for download

// ── startup ────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupDropzone();
    setupSlider();
});

async function checkHealth() {
    try {
        const res = await fetch(`${API}/health`);
        const data = await res.json();

        const dot = document.querySelector('.dot');
        const text = document.getElementById('status-text');

        if (data.model_loaded) {
            dot.classList.add('connected');
            text.textContent = `Model ready (${data.device})`;
        } else {
            text.textContent = 'Model not loaded';
        }
    } catch (e) {
        document.getElementById('status-text').textContent = 'API offline';
    }
}

// ── file handling ──────────────────────────────────────

function setupDropzone() {
    const zone = document.getElementById('dropzone');
    const input = document.getElementById('file-input');

    // drag events
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    // click to browse
    zone.addEventListener('click', () => input.click());
    input.addEventListener('change', () => {
        if (input.files.length) handleFile(input.files[0]);
    });
}

function handleFile(file) {
    if (!file.name.endsWith('.npy')) {
        alert('Please upload a .npy file');
        return;
    }

    selectedFile = file;
    document.getElementById('file-info').style.display = 'flex';
    document.getElementById('file-name').textContent = `${file.name} (${(file.size / 1024).toFixed(0)} KB)`;

    // try to parse the npy to show as preview
    parseNpy(file).then(arr => {
        inputVolume = arr;
        drawSlice('canvas-input', inputVolume, getSlice());
    }).catch(() => {
        // can't preview, that's fine — server will still process it
    });
}

// ── run segmentation ───────────────────────────────────

async function runSegmentation() {
    if (!selectedFile) return;

    const btn = document.getElementById('run-btn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');

    btn.disabled = true;
    loading.style.display = 'block';
    results.style.display = 'none';

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const res = await fetch(`${API}/predict`, { method: 'POST', body: formData });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || 'Prediction failed');
        }

        // get stats from headers
        const inferenceTime = res.headers.get('X-Inference-Time-Ms') || '?';
        const fgVoxels = res.headers.get('X-Foreground-Voxels') || '0';
        const totalVoxels = res.headers.get('X-Total-Voxels') || '1';

        // get mask as blob
        maskBlob = await res.blob();
        const arrayBuf = await maskBlob.arrayBuffer();
        maskVolume = parseNpyBuffer(arrayBuf);

        // show results
        showStats(inferenceTime, fgVoxels, totalVoxels);
        drawSlice('canvas-mask', maskVolume, getSlice());
        drawOverlay(getSlice());

        results.style.display = 'block';
    } catch (e) {
        alert(`Error: ${e.message}`);
    } finally {
        btn.disabled = false;
        loading.style.display = 'none';
    }
}

// ── stats ──────────────────────────────────────────────

function showStats(time, fg, total) {
    const pct = ((parseInt(fg) / parseInt(total)) * 100).toFixed(1);
    const grid = document.getElementById('stats-grid');
    grid.innerHTML = `
        <div class="stat">
            <div class="stat-value">${time}ms</div>
            <div class="stat-label">Inference Time</div>
        </div>
        <div class="stat">
            <div class="stat-value">${parseInt(fg).toLocaleString()}</div>
            <div class="stat-label">Tumor Voxels</div>
        </div>
        <div class="stat">
            <div class="stat-value">${pct}%</div>
            <div class="stat-label">Tumor Volume</div>
        </div>
        <div class="stat">
            <div class="stat-value">${parseInt(total).toLocaleString()}</div>
            <div class="stat-label">Total Voxels</div>
        </div>
    `;
}

// ── slice viewer ───────────────────────────────────────

function getSlice() {
    return parseInt(document.getElementById('slice-slider').value);
}

function setupSlider() {
    const slider = document.getElementById('slice-slider');
    const num = document.getElementById('slice-num');

    slider.addEventListener('input', () => {
        const s = parseInt(slider.value);
        num.textContent = s;

        if (inputVolume) drawSlice('canvas-input', inputVolume, s);
        if (maskVolume) {
            drawSlice('canvas-mask', maskVolume, s);
            drawOverlay(s);
        }
    });
}

function drawSlice(canvasId, volume, sliceIdx) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');

    if (!volume || volume.length === 0) return;

    // volume is flat typed array — figure out the 3D shape
    // assume shape is stored in volume.shape or infer from size
    const shape = volume.shape || inferShape(volume.length);
    const [d, h, w] = shape;

    const clampedSlice = Math.min(sliceIdx, d - 1);

    canvas.width = w;
    canvas.height = h;

    const imageData = ctx.createImageData(w, h);
    const offset = clampedSlice * h * w;

    for (let i = 0; i < h * w; i++) {
        let val = volume.data ? volume.data[offset + i] : volume[offset + i];

        // normalize to 0-255
        if (val <= 1.0) val = val * 255;
        val = Math.max(0, Math.min(255, val));

        const px = i * 4;
        imageData.data[px] = val;
        imageData.data[px + 1] = val;
        imageData.data[px + 2] = val;
        imageData.data[px + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);
}

function drawOverlay(sliceIdx) {
    if (!inputVolume || !maskVolume) return;

    const canvas = document.getElementById('canvas-overlay');
    const ctx = canvas.getContext('2d');

    const shape = inputVolume.shape || inferShape(inputVolume.length);
    const [d, h, w] = shape;

    const s = Math.min(sliceIdx, d - 1);
    canvas.width = w;
    canvas.height = h;

    const imageData = ctx.createImageData(w, h);
    const offset = s * h * w;

    const mShape = maskVolume.shape || inferShape(maskVolume.length);
    const mOffset = s * mShape[1] * mShape[2];

    for (let i = 0; i < h * w; i++) {
        let val = inputVolume.data ? inputVolume.data[offset + i] : inputVolume[offset + i];
        if (val <= 1.0) val = val * 255;
        val = Math.max(0, Math.min(255, val));

        let maskVal = maskVolume.data ? maskVolume.data[mOffset + i] : maskVolume[mOffset + i];

        const px = i * 4;
        if (maskVal > 0.5) {
            // red overlay for tumor
            imageData.data[px] = Math.min(255, val * 0.4 + 200);
            imageData.data[px + 1] = val * 0.3;
            imageData.data[px + 2] = val * 0.3;
        } else {
            imageData.data[px] = val;
            imageData.data[px + 1] = val;
            imageData.data[px + 2] = val;
        }
        imageData.data[px + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);
}

function inferShape(totalSize) {
    // try common shapes — 64x128x128, 64x256x256, etc
    const guesses = [
        [64, 128, 128],
        [64, 256, 256],
        [32, 128, 128],
        [128, 128, 128],
    ];
    for (const s of guesses) {
        if (s[0] * s[1] * s[2] === totalSize) return s;
    }
    // fallback: cube root
    const side = Math.round(Math.cbrt(totalSize));
    return [side, side, side];
}

// ── npy parsing ────────────────────────────────────────

async function parseNpy(file) {
    const buf = await file.arrayBuffer();
    return parseNpyBuffer(buf);
}

function parseNpyBuffer(buffer) {
    /**
     * Minimal .npy parser.
     * Reads the numpy header to get dtype and shape,
     * then returns a typed array with a .shape property.
     */
    const view = new DataView(buffer);

    // magic: \x93NUMPY
    const magic = String.fromCharCode(
        view.getUint8(0), view.getUint8(1), view.getUint8(2),
        view.getUint8(3), view.getUint8(4), view.getUint8(5)
    );
    if (!magic.includes('NUMPY')) {
        throw new Error('Not a valid .npy file');
    }

    const major = view.getUint8(6);
    const headerLen = (major >= 2)
        ? view.getUint32(8, true)
        : view.getUint16(8, true);

    const headerStart = (major >= 2) ? 12 : 10;
    const headerBytes = new Uint8Array(buffer, headerStart, headerLen);
    const header = new TextDecoder().decode(headerBytes);

    // parse shape from header like "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1, 64, 128, 128)}"
    const shapeMatch = header.match(/shape['"]\s*:\s*\(([^)]*)\)/);
    const descrMatch = header.match(/descr['"]\s*:\s*'([^']*)'/);

    const shape = shapeMatch
        ? shapeMatch[1].split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n))
        : [];

    const dtype = descrMatch ? descrMatch[1] : '<f4';
    const dataOffset = headerStart + headerLen;

    // pick typed array based on dtype
    let data;
    if (dtype.includes('f4')) {
        data = new Float32Array(buffer, dataOffset);
    } else if (dtype.includes('f8')) {
        data = new Float64Array(buffer, dataOffset);
    } else if (dtype.includes('u1') || dtype.includes('b')) {
        data = new Uint8Array(buffer, dataOffset);
    } else if (dtype.includes('i4')) {
        data = new Int32Array(buffer, dataOffset);
    } else {
        data = new Float32Array(buffer, dataOffset);
    }

    // strip batch + channel dims for visualization
    // e.g. (1, 1, 64, 128, 128) → [64, 128, 128]
    let vizShape = shape;
    while (vizShape.length > 3 && vizShape[0] === 1) {
        vizShape = vizShape.slice(1);
    }

    const result = { data, shape: vizShape, fullShape: shape, length: data.length };
    return result;
}

// ── download ───────────────────────────────────────────

function downloadMask() {
    if (!maskBlob) return;

    const a = document.createElement('a');
    a.href = URL.createObjectURL(maskBlob);
    a.download = 'segmentation.npy';
    a.click();
    URL.revokeObjectURL(a.href);
}
