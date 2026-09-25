import os
import shutil
from pathlib import Path

class OCR:
    def __init__(self, engine='easyocr', model_dir='models/ocr', tesseract_cmd=None):
        self.engine = engine
        if engine == 'easyocr':
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=str(model_dir),
                                         user_network_directory=str(Path(model_dir)/'user_network'), verbose=False)
        elif engine == 'tesseract':
            import pytesseract
            candidate = tesseract_cmd or os.environ.get('TESSERACT_CMD') or shutil.which('tesseract')
            if not candidate:
                usual = Path(r'C:\Program Files\Tesseract-OCR\tesseract.exe')
                if usual.exists():
                    candidate = str(usual)
            if not candidate:
                raise RuntimeError('Tesseract is not installed. Install it and set TESSERACT_CMD, or use --ocr easyocr.')
            pytesseract.pytesseract.tesseract_cmd = candidate
            pytesseract.get_tesseract_version()
            self.reader = pytesseract
        else:
            raise ValueError('OCR engine must be easyocr or tesseract')

    def read(self, image):
        if self.engine == 'tesseract':
            raw = self.reader.image_to_string(image, lang='eng', config='--oem 3 --psm 7')
            return raw, None, raw
        parts = self.reader.readtext(image, detail=1, paragraph=False,
                                     allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', workers=0)
        # OCR's text boxes only order text inside a model-produced plate crop.
        parts.sort(key=lambda p: (round(min(pt[1] for pt in p[0])/max(1,image.shape[0])*4), min(pt[0] for pt in p[0])))
        raw = '\n'.join(p[1] for p in parts)
        confidence = sum(float(p[2]) for p in parts)/len(parts) if parts else 0.0
        # Small state/dealer text is not part of the plate number. Keep the larger
        # text regions, while retaining every original OCR region in raw_text.
        heights=[max(pt[1] for pt in p[0])-min(pt[1] for pt in p[0]) for p in parts]
        cutoff=0.65*max(heights,default=0)
        number_text='\n'.join(p[1] for p,h in zip(parts,heights) if h>=cutoff)
        return raw, confidence, number_text
