# License plate detection and OCR

Give the program a vehicle image. YOLOv8 finds the plates, OpenCV crops each box with 8% padding, and OCR reads the text. All crops come from model predictions. No coordinates or answers are supplied during inference.

## Run

Use Python 3.11. Run these commands from this folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py --image data/images/JRV1942.jpg --format br
```

The command prints the boxes, raw OCR text and cleaned text, and saves `prediction.jpg`. EasyOCR downloads its English models on first use. The trained detector is included in `models/plate.pt`.

On Linux/macOS, activate the environment with `source .venv/bin/activate` instead.

For an unseen Azerbaijani plate:

```bash
python run.py --image car.jpg --format az
```

`az` checks the ordinary `12-AB-345` format; `br` checks the older Brazilian `ABC-1234` format used in the test data. Spaces and punctuation are removed. Confusable characters such as `O/0` and `I/1` are corrected only at the expected letter/digit positions. Other countries can use `--format generic`. Validation checks the shape of the string, not whether a plate was officially issued.

EasyOCR's smaller state/dealer text regions are excluded from the number candidate. The complete raw text is still saved. Ambiguous or missing characters are not filled from a list of known answers.

Python use:

```python
from plate_pipeline import PlatePipeline

pipeline = PlatePipeline(plate_format="az")
texts = pipeline("car.jpg")  # one string per detection; [] if none
details = pipeline.predict("car.jpg")  # boxes, raw text and confidence
```

## Results

On 30 held-out images: mean IoU **0.846**, detection precision/recall at IoU 0.5 **100%/100%**, EasyOCR exact match **23/30 (76.7%)**, CER **7.14%**. These figures describe this small Brazilian test set, not every country or road condition.

See [test results](reports/results.md), [failure examples](reports/failures.md) and [multiple-vehicle checks](reports/cases.md). The CSV reports contain one row per ground-truth plate plus any extra detections.

Re-run the same comparison:

```bash
python evaluate.py
```

Tesseract is required for the comparison. Install the [Windows build](https://github.com/UB-Mannheim/tesseract/wiki), then pass its executable if it is not on PATH:

```bash
python evaluate.py --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

To evaluate only EasyOCR, use `python evaluate.py --engines easyocr`.

## Bonuses

```bash
# Webcam; press q to close
python run.py --video 0 --format az --db parking.sqlite3

# Dashcam file; save boxes and plate text in an output video
python run.py --video drive.mp4 --format az --output annotated.mp4 --headless

# Record an image result with a UTC timestamp
python run.py --image car.jpg --format az --db parking.sqlite3
```

The SQLite `sightings` table stores timestamp, source, plate, box and confidence. Invalid or empty strings are skipped; repeats from the same source have a 30-second cooldown. The live loop displays its measured processing FPS. Speed depends on the CPU and the number of plates.

## Data and training

The [OpenALPR benchmark](https://github.com/openalpr/benchmarks) supplies the image annotations: pixel boxes and plate strings. These are the source dataset's labels, not newly claimed personal annotations. The 114 Brazilian images are split by a fixed hash of their filenames: 74 train, 10 validation, 30 test. Plate identities and file checksums are checked for overlap. The exact upstream revision, original coordinates and image hashes are in `data/manifest.json`.

One source training box starts at x = -1 (`PJT2905`); its working coordinate is clipped to zero. The original is retained as `source_box`. Test annotations were visually checked; no model outputs were used to create them.

YOLOv8n starts from COCO weights and is fine-tuned for one class, `license_plate`, at image size 416. It is small enough to run on CPU. Checkpoint selection uses validation data, not the test set. The ordinary COCO detector is not used as a plate detector.

To train again:

```bash
python train.py --epochs 25
```

Use `--serial-scan` if Windows blocks named pipes during label scanning. YOLO label files are generated from the manifest, so the pixel-to-normalized-coordinate conversion is reproducible. Training history is saved in `reports/training/`.

## Checks and limits

```bash
python -m pytest -q
python check_cases.py
```

The detector was trained on a small Brazilian dataset. Azerbaijani format cleaning is implemented, but accuracy on Azerbaijani roads is not measured. Blur, strong angles and occlusion can still make text unreadable. Invalid OCR output is returned visibly rather than replaced with a guessed plate. The failure report shows actual errors.

On this Windows machine, the Tesseract installer did not launch. Its files were extracted locally and its executable was passed explicitly. EasyOCR was the pip-only alternative. No system-wide PATH changes are necessary.

References: [Ultralytics training](https://docs.ultralytics.com/modes/train/), [EasyOCR](https://github.com/JaidedAI/EasyOCR), [Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html). Upstream dataset licensing is included in `data/LICENSE`; the project uses AGPL-3.0.
