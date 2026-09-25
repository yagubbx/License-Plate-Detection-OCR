# License Plate Detection & OCR

YOLOv8 finds a plate, OpenCV crops it with padding, and EasyOCR reads the text. Detection boxes always come from the model.

## Run

Use Python 3.11. Open a terminal in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py --image data/images/car_1.png --format az
```

The terminal shows a result card with the plate text, detection score, raw OCR, format status and saved image path. The annotated image is saved as `prediction.jpg`. Use `--output result.png` to choose another output. Add `--json` if another program needs JSON.

The detector is included. EasyOCR downloads its English weights on first use. No retraining is needed.

## The 25 images

- `car_1.png`–`car_18.png`: the supplied Azerbaijani photos; use `--format az`.
- `car_19.png`–`car_25.png`: seven retained Brazilian photos; use `--format br`.

All are real PNG files. The filename is never used to infer the plate number. `data/manifest.json` records paths, original filenames, dimensions, checksums, plate strings and evaluation boxes. The 18 supplied photos were visually annotated before inference; the seven retained photos preserve their source annotations. There are 26 labeled plates because `car_2` also has a legible background plate.

These 25 images are test-only. The detector was previously fine-tuned from COCO YOLOv8n on 74 different Brazilian images, with checkpoint selection on a separate 10-image validation set. That old dataset is not included. The model has not been fine-tuned on the supplied Azerbaijani photos. The annotated boxes are used only for evaluation, never for inference cropping.

## Evaluation

```powershell
.\.venv\Scripts\python.exe evaluate.py --engines easyocr
.\.venv\Scripts\python.exe -m pytest -q
```

[Results](reports/results.md) · [Failure examples](reports/failures.md) · [Multiple-vehicle checks](reports/cases.md)

To compare both OCR engines on identical model crops, install [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki), then run:

```powershell
.\.venv\Scripts\python.exe evaluate.py --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

EasyOCR is the pip-only option. During development, the Tesseract installer did not launch in the restricted Windows environment; its files were extracted locally and the executable path was passed explicitly.

## Bonuses

```powershell
# Webcam: live boxes; q closes the window
.\.venv\Scripts\python.exe run.py --video 0 --format az --db parking.sqlite3

# Video file
.\.venv\Scripts\python.exe run.py --video drive.mp4 --format az --output annotated.mp4 --headless
```

The SQLite log records plate, UTC timestamp, source, box and detection score. Empty/invalid readings are skipped; repeated plates from the same source have a 30-second cooldown. Live processing speed depends on hardware.

Regex validation supports ordinary Azerbaijani `12-AB-345` and older Brazilian `ABC-1234` formats. Position-based `O/0`, `I/1` and similar corrections remove common OCR noise. `--format generic` disables country-specific correction. A valid format or high model score does not prove the reading is correct.

Python use:

```python
from plate_pipeline import PlatePipeline
pipeline = PlatePipeline(plate_format="az")
texts = pipeline("data/images/car_1.png")  # [] if nothing is detected
```

Sources: [Ultralytics](https://docs.ultralytics.com/modes/train/), [EasyOCR](https://github.com/JaidedAI/EasyOCR), [OpenALPR source images](https://github.com/openalpr/benchmarks). The source license for the seven retained images is in `data/SOURCE_LICENSE`. Supplied photos retain their original rights. Project code uses AGPL-3.0.
