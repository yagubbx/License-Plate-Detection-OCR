# License Plate Detection & OCR

This project finds and reads Azerbaijani license plates.

YOLOv8 finds the plate. OpenCV crops and straightens it. EasyOCR reads the text. Simple rules check the `00-XX-000` format and fix common letter/digit errors.

**[Start here](start.md)** · **[Test report and failures](reports/report.md)**

## What is included?

- 58 photos: 34 for development and 24 for testing.
- Model-based detection, padded crops and support for multiple plates.
- Terminal results and a saved image with boxes.
- Two-line text reading, format checks, video input and a parking log.

The test result is **19/25 readable plates correct (76%)**, including background plates. Mean IoU is **0.7606**. The report explains the errors. The three two-line checks use synthetic crops; real two-line photos have not been tested.

EasyOCR is the default. **You do not need Tesseract.** Tesseract is a second OCR tool kept only for the optional comparison bonus.

The detector is included. EasyOCR downloads its models on first use. No training is needed. See [model details](models/README.md) for the source and limits.

Python use:

```python
from plate_pipeline import PlatePipeline

read_plate = PlatePipeline()
print(read_plate("data/images/car_1.png"))
```

Boxes come from the model. Image names and test labels are never used to predict text. Use any filename for a new image; labels are needed only for evaluation.

Code: AGPL-3.0. Photos keep their original rights.
