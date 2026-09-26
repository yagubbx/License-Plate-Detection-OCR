import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from .geometry import prepare_crop, straighten_plate
from .text import clean_text, valid_plate, normalize

ALPHABET='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def ordered_text(parts):
    """Group OCR regions into rows, then read each row left to right."""
    if not parts: return '',0.0,''
    height=lambda p:max(pt[1] for pt in p[0])-min(pt[1] for pt in p[0])
    center=lambda p:sum(pt[1] for pt in p[0])/len(p[0])
    largest=max(height(p) for p in parts)
    rows=[]
    for part in sorted(parts,key=center):
        row=next((row for row in rows if abs(center(part)-np.mean([center(p) for p in row]))<.6*largest),None)
        if row is None: rows.append([part])
        else: row.append(part)
    for row in rows: row.sort(key=lambda p:min(pt[0] for pt in p[0]))
    raw='\n'.join(' '.join(p[1] for p in row) for row in rows)
    kept=[[p for p in row if height(p)>=.6*largest and normalize(p[1])!='AZ'] for row in rows]
    number='\n'.join(''.join(p[1] for p in row) for row in kept if row)
    scores=[float(p[2]) for row in kept for p in row]
    return raw,sum(scores)/len(scores) if scores else 0.0,number

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

    def read(self, image, plate_format='az'):
        gray=prepare_crop(image) if image.ndim==3 else image
        rect=straighten_plate(image)
        if self.engine == 'tesseract':
            image=rect if rect is not None else gray
            mode=6 if image.shape[1]/image.shape[0]<2.7 else 7
            raw = self.reader.image_to_string(image, lang='eng', config=f'--oem 3 --psm {mode}')
            return raw, None, raw
        variants=[('original',gray)]
        if rect is not None:
            variants.append(('straight',rect))
            binary=cv2.threshold(rect,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            variants.append(('straight_binary',binary))
        attempts=[]
        for name,img in variants:
            parts=self.reader.readtext(img,detail=1,paragraph=False,allowlist=ALPHABET,workers=0)
            raw,confidence,number=ordered_text(parts)
            attempts.append((raw,confidence,number,name))
            if img.shape[1]/img.shape[0]>=2.7:
                regions=[img]
                if plate_format=='az':
                    h,w=img.shape
                    regions.append(img[int(.08*h):int(.92*h),int(.13*w):int(.98*w)])
                for region in regions:
                    parts=self.reader.recognize(region,detail=1,allowlist=ALPHABET,decoder='beamsearch',beamWidth=5)
                    raw,confidence,number=ordered_text(parts)
                    attempts.append((raw,confidence,number,name))
            else:
                # Compact plates: read the two halves separately if text detection
                # could not join them. These are relative crop regions, not GT boxes.
                h,w=img.shape
                rows=[img[:h//2,:],img[h//2:,:]]
                texts=[];scores=[]
                for row in rows:
                    parts=self.reader.readtext(row,detail=1,paragraph=False,allowlist=ALPHABET,workers=0)
                    raw,score,number=ordered_text(parts);texts.append(number);scores.append(score)
                attempts.append(('\n'.join(texts),sum(scores)/2,'\n'.join(texts),name))
        return self.choose(attempts,plate_format)

    @staticmethod
    def choose(attempts,plate_format):
        cleaned=[clean_text(a[2],plate_format) for a in attempts]
        def score(i):
            a=attempts[i];text=cleaned[i]
            agreement=sum(other==text for other in cleaned)-1
            return (bool(valid_plate(text,plate_format)), a[1]+.15*min(agreement,3)+(.1 if a[3].startswith('straight') else 0))
        best=max(range(len(attempts)),key=score)
        return attempts[best][:3]
