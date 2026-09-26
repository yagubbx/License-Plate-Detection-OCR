# License Plate Detection & OCR

YOLOv8 locates the plate. OpenCV adds crop padding and straightens a visible plate border. EasyOCR reads the crop, then position-based rules clean ordinary Azerbaijani `00-XX-000` plates. Detection never reads the annotation file or uses the image name to guess a number.

## Start

Install Python 3.11. Open PowerShell **in this folder**, then run these commands one at a time:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py --image data/images/car_1.png
```

The detector is included. EasyOCR downloads its English models on the first run; later runs use the local cache. No Tesseract installation or training is needed for the default command.

The terminal shows the plate as `90-BG-028`, scores, raw OCR and the saved image path. Open `prediction.jpg` to see the detected box. JSON is optional:

```powershell
.\.venv\Scripts\python.exe run.py --image data/images/car_7.png --output result.png
.\.venv\Scripts\python.exe run.py --image data/images/car_7.png --json
```

Use any filename for a new picture. Updating `data/manifest.json` is necessary only when including that picture in evaluation, not for prediction.

## Data and results

The supplied archive contains **58 photos**, renamed `car_1.png` through `car_58.png` and converted to real PNG files. It replaces the previous collection. Ground truth contains 68 manually checked plate boxes: 60 readable strings and eight unreadable/partial background plates with `text: null`.

- **34 development images:** used to choose settings and OCR rules.
- **24 held-out test images:** reserved before OCR tuning. Previously used photos stay in development; photos of the same primary plate stay in the same split.

All 58 primary plates were detected. EasyOCR read **53/58 (91.4%)** primary plates correctly across the complete collection. That includes development images and is **not** the held-out accuracy. On the 24 held-out images, primary-plate accuracy was **19/24 (79.2%)**. Including the readable background plate, exact match was **19/25 (76.0%)**, CER **0.0914**, and mean IoU across all 30 test boxes **0.7606**. The requested one-or-two-error target has not been reached.

```powershell
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe -m pytest -q
```

[Test report](reports/results.md) · [All 58 images](reports/all_images.csv) · [Failure examples](reports/failures.md) · [Crop and layout checks](reports/cases.md)

The CSV reports preserve the raw OCR, predicted and padded boxes, corrected text and success/failure for every annotated plate. Missed plates count as failures. Unreadable background strings count in detection, not OCR scores. The annotation scope excludes distant plates smaller than 25×8 pixels. There is no per-image correction list.

The configured pretrained detector was selected because the previous model missed Azerbaijani plates. See [model source and limitations](models/README.md). These supplied photos were not used to train model weights. Upstream training-image overlap cannot be verified.

## OCR and two-line plates

OCR tries the padded crop, its automatically straightened version and a thresholded version. It selects a reading using format, confidence and agreement. Character substitutions apply only to the expected digit/letter positions; for example `O` becomes `0` in a digit position, and the open `4` glyph can be read as `L`. These rules cannot reliably fix a `4` read as `2`, or restore missing characters.

Text regions are ordered top-to-bottom and left-to-right for two-line plates. All 58 supplied primary plates are single-line. Three synthetic two-line crops passed OCR checks; performance on real two-line vehicle photos has **not** been established.

Multiple detections produce separate crops and results; no detections produce an empty list. A matching regex or high confidence is not proof that the plate is correct.

## Bonuses

Regex validation, a timestamped SQLite parking log, video/webcam input, and an optional Tesseract comparison are included.

```powershell
# Image with a parking log
.\.venv\Scripts\python.exe run.py --image data/images/car_1.png --db parking.sqlite3

# Webcam; press q to close
.\.venv\Scripts\python.exe run.py --video 0 --db parking.sqlite3

# Video file
.\.venv\Scripts\python.exe run.py --video drive.mp4 --output annotated.mp4 --headless
```

The log stores valid readings with UTC timestamps and skips repeated plates from the same source for 30 seconds. Video shows live boxes; CPU processing may be slower than the camera frame rate.

To compare Tesseract with EasyOCR on identical detector boxes, install [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki), then run:

```powershell
.\.venv\Scripts\python.exe evaluate.py --engines easyocr tesseract --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

The included comparison used Tesseract 5.4.0. Its installer could not run in the development environment, so its files were extracted locally and the executable path was passed explicitly. EasyOCR remains the pip-only default. The default evaluation needs no Tesseract.

Python use:

```python
from plate_pipeline import PlatePipeline

pipeline = PlatePipeline()
texts = pipeline("data/images/car_1.png")  # ['90BG028']; [] if no plate is found
```

`--format generic` disables country-specific cleaning. `--format br` retains the earlier Brazilian-format option, but the included photos and current evaluation focus on Azerbaijan.

Sources: [Ultralytics](https://docs.ultralytics.com/quickstart/), [EasyOCR](https://github.com/JaidedAI/EasyOCR), [detector model](https://huggingface.co/Koushim/yolov8-license-plate-detection). Project code is AGPL-3.0. Supplied photos retain their original rights.
