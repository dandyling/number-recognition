# 🔢 Number Recognition (0–9)

A small end-to-end project that trains a convolutional neural network to
recognise handwritten digits and serves it behind a web UI where you can
**upload an image and get the predicted number**.

- **Model** — a compact CNN trained on the [MNIST](http://yann.lecun.com/exdb/mnist/)
  dataset (downloaded from the internet), reaching **~99% test accuracy**.
- **Backend** — a Flask API that loads the trained weights and predicts the
  digit from an uploaded image.
- **Frontend** — a single page where you upload an image and see the digit and
  the model's confidence.
- **E2E test** — drives a real Chromium browser with Playwright, uploads sample
  images, asserts the predictions, and records a video of the whole flow.

## Project layout

```
model/train.py          CNN definition + training loop (saves model/mnist_cnn.pt)
model/mnist_cnn.pt       Trained weights (committed so the app runs out of the box)
backend/app.py           Flask server: GET / (UI) and POST /predict
frontend/index.html      Upload UI
e2e/test_e2e.py          Playwright end-to-end test that records a video
e2e/samples/             Sample digit images used by the test
e2e/videos/              Recorded run -> digit_recognition_e2e.mp4
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

## End-to-end test (with video)

```bash
# with the server running on http://localhost:5000
python e2e/test_e2e.py
```

It launches Chromium, uploads `e2e/samples/*.png`, verifies each prediction is
correct, and writes `e2e/videos/digit_recognition_e2e.mp4`.

Latest run:

```
mnist_7.png -> 7  (99.9%)
typed_5.png -> 5  (100.0%)   # black digit on white background
mnist_3.png -> 3  (75.0%)
E2E PASSED: all uploaded digits were recognised correctly.
```
