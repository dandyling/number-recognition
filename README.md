# 🔢 Number Recognition (0–9)

A small end-to-end project that trains a convolutional neural network to
recognise handwritten digits and serves it behind a web UI where you can
**upload an image and get the predicted number**.

- **Model** — a compact CNN trained on the [MNIST](http://yann.lecun.com/exdb/mnist/)
  dataset (downloaded from the internet), reaching **~99% test accuracy**.
- **Static web app** (`public/`) — the model exported to ONNX and run **entirely
  in the browser** with onnxruntime-web. This is what's deployed to **Vercel**:
  no server, no cold starts, scales as a plain static site.
- **Flask backend** (`backend/`) — a reference API that loads the PyTorch weights
  and predicts from an uploaded image. Handy for local development.
- **E2E tests** — drive a real Chromium browser with Playwright, upload sample
  images, assert the predictions, and record a video of the whole flow.

The static app and the Flask app use the **same model and the same
preprocessing**, so they produce identical predictions.

## Project layout

```
model/train.py           CNN definition + training loop (saves model/mnist_cnn.pt)
model/mnist_cnn.pt        Trained PyTorch weights
model/mnist_cnn.onnx      Model exported to ONNX (used by the browser app)
public/index.html         Static upload UI deployed to Vercel
public/app.js             In-browser preprocessing + ONNX inference
public/ort/               Vendored onnxruntime-web runtime (js + wasm)
public/mnist_cnn.onnx     Model served to the browser
backend/app.py            Flask server: GET / (UI) and POST /predict
frontend/index.html       Upload UI for the Flask app
e2e/test_e2e_static.py    Playwright E2E against the static (ONNX) app
e2e/test_e2e.py           Playwright E2E against the Flask app
e2e/samples/              Sample digit images used by the tests
e2e/videos/               Recorded runs (.webm / .mp4)
vercel.json               Static deploy config (serves public/)
```

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # see note in requirements.txt for the torch CPU index

# (optional) re-train the model — weights are already committed
python model/train.py

# run the web app
python backend/app.py
# open http://localhost:5000 and upload an image of a digit
```

## How an uploaded image is processed

Uploaded photos rarely look like raw MNIST samples, so `backend/app.py`:

1. flattens transparency and converts to grayscale,
2. detects the background from the border pixels and inverts dark-on-light
   images so the digit is white on black (MNIST convention),
3. crops to the digit's bounding box and centres it in a 28×28 frame,
4. normalises with the MNIST mean/std the model was trained on.

This lets it handle both real MNIST-style images and, e.g., a black digit typed
on a white background.

## End-to-end tests (with video)

```bash
# static / ONNX app (what's deployed to Vercel)
cd public && python -m http.server 8000 &
BASE_URL=http://localhost:8000 python e2e/test_e2e_static.py

# Flask app
python backend/app.py &
BASE_URL=http://localhost:5000 python e2e/test_e2e.py
```

Each launches Chromium, uploads `e2e/samples/*.png`, verifies the prediction,
and records a video to `e2e/videos/`.

Latest run (identical for both stacks):

```
mnist_7.png -> 7  (99.9%)
typed_5.png -> 5  (100.0%)   # black digit on white background
mnist_3.png -> 3  (75.0%)
E2E PASSED: all uploaded digits were recognised correctly.
```

## Deploy to Vercel

The static app is deployed to Vercel under the *dandyling's projects* team.
`vercel.json` serves `public/` as a static site and `.vercelignore` keeps the
Python `venv/` and the MNIST dataset out of the upload. Since inference runs
client-side in WebAssembly, there's no backend to host.
