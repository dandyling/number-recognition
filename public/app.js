// Client-side digit recognition using onnxruntime-web.
// The model and preprocessing mirror the PyTorch training pipeline so that the
// browser produces the same predictions the server-side model would.

const MNIST_MEAN = 0.1307;
const MNIST_STD = 0.3081;

// onnxruntime-web is loaded via a classic <script> tag, exposing `ort`.
ort.env.wasm.wasmPaths = "/ort/";
ort.env.wasm.numThreads = 1; // single-threaded: no SharedArrayBuffer / COOP-COEP needed

let session = null;
const sessionReady = ort.InferenceSession.create("/mnist_cnn.onnx").then((s) => {
  session = s;
});

/**
 * Convert an <img>/<canvas> source into a normalised 28x28 Float32 tensor,
 * matching backend/app.py:preprocess().
 */
function preprocess(source) {
  // 1) Draw onto a canvas with a white background (flattens transparency).
  const w = source.naturalWidth || source.width;
  const h = source.naturalHeight || source.height;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(source, 0, 0, w, h);
  const { data } = ctx.getImageData(0, 0, w, h);

  // 2) Grayscale (luminance).
  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }

  // 3) Decide polarity from the border pixels (assumed background).
  let borderSum = 0, borderCount = 0;
  for (let x = 0; x < w; x++) {
    borderSum += gray[x] + gray[(h - 1) * w + x];
    borderCount += 2;
  }
  for (let y = 0; y < h; y++) {
    borderSum += gray[y * w] + gray[y * w + (w - 1)];
    borderCount += 2;
  }
  const invert = borderSum / borderCount > 127; // light bg -> invert to white-on-black
  for (let i = 0; i < w * h; i++) {
    if (invert) gray[i] = 255 - gray[i];
  }

  // 4) Crop to the digit's bounding box (pixels > 30).
  let minX = w, minY = h, maxX = -1, maxY = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (gray[y * w + x] > 30) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) {
    // blank image: fall back to the whole frame
    minX = 0; minY = 0; maxX = w - 1; maxY = h - 1;
  }
  const cropW = maxX - minX + 1;
  const cropH = maxY - minY + 1;

  // Put the cropped digit (white-on-black) onto a canvas to rescale it.
  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = cropW;
  cropCanvas.height = cropH;
  const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });
  const cropImg = cropCtx.createImageData(cropW, cropH);
  for (let y = 0; y < cropH; y++) {
    for (let x = 0; x < cropW; x++) {
      const v = gray[(minY + y) * w + (minX + x)];
      const di = (y * cropW + x) * 4;
      cropImg.data[di] = v;
      cropImg.data[di + 1] = v;
      cropImg.data[di + 2] = v;
      cropImg.data[di + 3] = 255;
    }
  }
  cropCtx.putImageData(cropImg, 0, 0);

  // 5) Scale to fit in a 20x20 box (preserve aspect), centre in a 28x28 frame.
  const scale = Math.min(20 / cropW, 20 / cropH);
  const newW = Math.max(1, Math.round(cropW * scale));
  const newH = Math.max(1, Math.round(cropH * scale));
  const frame = document.createElement("canvas");
  frame.width = 28;
  frame.height = 28;
  const fctx = frame.getContext("2d", { willReadFrequently: true });
  fctx.fillStyle = "#000000";
  fctx.fillRect(0, 0, 28, 28);
  fctx.imageSmoothingEnabled = true;
  fctx.imageSmoothingQuality = "high";
  const offX = Math.floor((28 - newW) / 2);
  const offY = Math.floor((28 - newH) / 2);
  fctx.drawImage(cropCanvas, offX, offY, newW, newH);

  // 6) Read back, normalise, build the [1,1,28,28] tensor.
  const final = fctx.getImageData(0, 0, 28, 28).data;
  const input = new Float32Array(28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    const v = final[i * 4] / 255.0; // grayscale: R=G=B
    input[i] = (v - MNIST_MEAN) / MNIST_STD;
  }
  return new ort.Tensor("float32", input, [1, 1, 28, 28]);
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

async function predict(source) {
  await sessionReady;
  const tensor = preprocess(source);
  const output = await session.run({ input: tensor });
  const logits = Array.from(output.logits.data);
  const probs = softmax(logits);
  let digit = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[digit]) digit = i;
  return { digit, confidence: probs[digit], probabilities: probs };
}

// ---- UI wiring ---------------------------------------------------------------

const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const dropText = document.getElementById("dropText");
const preview = document.getElementById("preview");
const predictBtn = document.getElementById("predictBtn");
const result = document.getElementById("result");
const digitEl = document.getElementById("digit");
const confidenceEl = document.getElementById("confidence");
const errorEl = document.getElementById("error");

let selectedFile = null;

function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  dropText.textContent = file.name;
  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
  predictBtn.disabled = false;
  result.style.display = "none";
  errorEl.style.display = "none";
}

fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  })
);
dropzone.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  predictBtn.disabled = true;
  predictBtn.textContent = "Recognizing…";
  errorEl.style.display = "none";
  try {
    const img = new Image();
    img.src = URL.createObjectURL(selectedFile);
    await img.decode();
    const data = await predict(img);
    digitEl.textContent = data.digit;
    confidenceEl.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    result.style.display = "block";
  } catch (err) {
    errorEl.textContent = err.message || String(err);
    errorEl.style.display = "block";
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Recognize digit";
  }
});
