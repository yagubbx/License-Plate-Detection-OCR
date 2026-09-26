from pathlib import Path
import os
import cv2
import numpy as np
from .geometry import padded_crop, prepare_crop
from .ocr import OCR
from .text import clean_text, valid_plate, display_plate

for name, directory in [('YOLO_CONFIG_DIR', '.cache/ultralytics'), ('MPLCONFIGDIR', '.cache/matplotlib')]:
    os.environ.setdefault(name, str(Path(directory).resolve()))
    Path(os.environ[name]).mkdir(parents=True, exist_ok=True)

class PlatePipeline:
    def __init__(self, weights='models/plate.pt', ocr='easyocr', plate_format='az', confidence=0.25,
                 padding=0.08, tesseract_cmd=None, model_dir=None):
        from ultralytics import YOLO
        if not Path(weights).is_file():
            raise FileNotFoundError(f'Missing detector: {weights}. Restore models/plate.pt from the project archive.')
        if not 0 <= confidence <= 1:
            raise ValueError('confidence must be between 0 and 1')
        clean_text('', plate_format)
        self.model = YOLO(str(weights))
        self.plate_classes = [int(k) for k,v in self.model.names.items() if 'plate' in v.lower()]
        if not self.plate_classes:
            raise ValueError('Weights have no license-plate class; ordinary COCO weights cannot detect plates.')
        self.ocr = OCR(ocr, model_dir or os.environ.get('PLATE_OCR_DIR','models/ocr'), tesseract_cmd)
        self.image_size = 960
        self.plate_format, self.confidence, self.padding = plate_format, confidence, padding

    @staticmethod
    def load_image(image):
        if isinstance(image, (str,Path)):
            path=Path(image)
            if not path.is_file():
                raise FileNotFoundError(path)
            image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None or not isinstance(image,np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError('Expected a readable image path or a nonempty BGR image')
        return image

    def detect(self, image):
        result=self.model.predict(image, conf=self.confidence, classes=self.plate_classes,
                                  imgsz=self.image_size, iou=0.5, device='cpu', verbose=False)[0]
        return [{'box':list(map(float,b.xyxy[0].tolist())), 'confidence':float(b.conf.item())}
                for b in result.boxes]

    def recognize(self, image, detection):
        crop,bounds=padded_crop(image,detection['box'],self.padding)
        raw,confidence,number_text=self.ocr.read(crop,self.plate_format)
        text=clean_text(number_text,self.plate_format)
        return {**detection, 'crop_box':bounds, 'raw_text':raw, 'text':text,'display_text':display_plate(text,self.plate_format),
                'ocr_text':number_text, 'ocr_confidence':confidence, 'format_valid':valid_plate(text,self.plate_format)}

    def predict(self, image):
        image=self.load_image(image)
        return [self.recognize(image,d) for d in self.detect(image)]

    def __call__(self, image):
        """Image in, list of plate strings out; [] means no detections."""
        return [p['text'] for p in self.predict(image)]

def draw_predictions(image, predictions):
    canvas=image.copy()
    for p in predictions:
        x1,y1,x2,y2=map(round,p['box'])
        cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,210,0),2)
        cv2.putText(canvas,p.get('display_text',p['text']) or '(unreadable)',(x1,max(18,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,210,0),2)
    return canvas
