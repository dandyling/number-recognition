"""End-to-end test for the static (client-side, ONNX) digit recognizer.

Launches a real Chromium browser against the static build, uploads sample digit
images through the UI, and asserts the predicted digit shown on the page is
correct. The whole session is recorded to a video.
"""

import os
import sys
import time

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "samples")
VIDEO_DIR = os.path.join(HERE, "videos")
CHROME = "/opt/pw-browsers/chromium-1223/chrome-linux64/chrome"
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")

CASES = [
    ("mnist_7.png", 7),
    ("typed_5.png", 5),
    ("mnist_3.png", 3),
]


def run():
    os.makedirs(VIDEO_DIR, exist_ok=True)
    failures = []
    console_errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=CHROME)
        context = browser.new_context(
            viewport={"width": 900, "height": 800},
            record_video_dir=VIDEO_DIR,
            record_video_size={"width": 900, "height": 800},
        )
        page = context.new_page()
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)

        page.goto(BASE_URL)
        page.wait_for_selector("h1")
        print(f"Loaded page: {page.title()!r}")
        time.sleep(1.0)

        for filename, expected in CASES:
            sample = os.path.join(SAMPLES, filename)
            print(f"\n--- Uploading {filename} (expecting {expected}) ---")

            page.set_input_files("#fileInput", sample)
            time.sleep(0.8)

            page.click("#predictBtn")
            page.wait_for_selector("#result", state="visible", timeout=30000)
            page.wait_for_function(
                "document.getElementById('digit').textContent.trim() !== '-'"
                " && document.getElementById('digit').textContent.trim() !== '–'",
                timeout=30000,
            )
            time.sleep(0.8)

            predicted = page.inner_text("#digit").strip()
            confidence = page.inner_text("#confidence").strip()
            print(f"Predicted: {predicted}  ({confidence})")

            if predicted != str(expected):
                failures.append(f"{filename}: expected {expected}, got {predicted}")

        context.close()
        browser.close()

    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".webm")]
    if videos:
        newest = max(videos, key=lambda f: os.path.getmtime(os.path.join(VIDEO_DIR, f)))
        final = os.path.join(VIDEO_DIR, "digit_recognition_vercel_e2e.webm")
        if newest != os.path.basename(final):
            os.replace(os.path.join(VIDEO_DIR, newest), final)
        print(f"\nVideo saved to: {final}")

    if console_errors:
        print("\nConsole errors:")
        for e in console_errors:
            print("  -", e)

    if failures:
        print("\nE2E FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)

    print("\nE2E PASSED: all uploaded digits were recognised correctly (client-side ONNX).")


if __name__ == "__main__":
    run()
