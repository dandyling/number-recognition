"""Flask backend that serves the digit-recognition model.

Exposes:
- ``GET  /``        -> the single-page frontend
- ``POST /predict`` -> accepts an uploaded image, returns the predicted digit
                       (0-9) together with a confidence score and the full
                       probability distribution.
"""

import io
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps

# Make the model package importable so we can reuse the exact Net definition
# and normalisation constants the model was trained with.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.train import MNIST_MEAN, MNIST_STD, Net, WEIGHTS_PATH  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(HERE, "..", "frontend")

app = Flask(__name__, static_folder=None)

device = torch.device("cpu")
model = Net().to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()


def preprocess(image_bytes):
    """Turn an arbitrary uploaded image into a normalised 28x28 MNIST tensor.

    Handles colour images, transparency, and either polarity (dark-on-light or
    light-on-dark). MNIST digits are white strokes on a black background, so we
    detect the background from the image corners and invert when necessary.
    """
    img = Image.open(io.BytesIO(image_bytes))

    # Flatten transparency onto a white background, then go grayscale.
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    img = img.convert("L")

    arr = np.asarray(img, dtype=np.float32)

    # Decide polarity from the border pixels (assumed to be background).
    border = np.concatenate(
        [arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]]
    )
    if border.mean() > 127:  # light background -> invert to white-on-black
        img = ImageOps.invert(img)
        arr = np.asarray(img, dtype=np.float32)

    # Crop to the digit's bounding box so framing/scale matches MNIST.
    mask = arr > 30
    if mask.any():
        ys, xs = np.where(mask)
        img = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

    # MNIST normalises digits into a 20x20 box centred in a 28x28 frame.
    img = ImageOps.contain(img, (20, 20), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    offset = ((28 - img.width) // 2, (28 - img.height) // 2)
    canvas.paste(img, offset)

    tensor = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0)
    tensor = (tensor - MNIST_MEAN) / MNIST_STD
    return tensor.view(1, 1, 28, 28)


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image file provided"}), 400

    file = request.files["image"]
    try:
        tensor = preprocess(file.read())
    except Exception as exc:  # noqa: BLE001 - report any decode/processing error
        return jsonify({"error": f"could not process image: {exc}"}), 400

    with torch.no_grad():
        logits = model(tensor.to(device))
        probs = F.softmax(logits, dim=1)[0]
        digit = int(probs.argmax().item())
        confidence = float(probs[digit].item())

    return jsonify(
        {
            "digit": digit,
            "confidence": confidence,
            "probabilities": [float(p) for p in probs],
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
