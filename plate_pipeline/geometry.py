import math
import cv2

def iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, right-left) * max(0, bottom-top)
    union = max(0,a[2]-a[0])*max(0,a[3]-a[1]) + max(0,b[2]-b[0])*max(0,b[3]-b[1]) - overlap
    return overlap / union if union else 0.0

def padded_crop(image, box, padding=0.08):
    if padding < 0:
        raise ValueError('Padding must be nonnegative')
    h,w = image.shape[:2]
    x1,y1,x2,y2 = box
    dx,dy = (x2-x1)*padding, (y2-y1)*padding
    bounds = [max(0,min(w,math.floor(x1-dx))), max(0,min(h,math.floor(y1-dy))),
              max(0,min(w,math.ceil(x2+dx))), max(0,min(h,math.ceil(y2+dy)))]
    a,b,c,d = bounds
    if c <= a or d <= b:
        raise ValueError('Empty detection box after clipping')
    return image[b:d,a:c].copy(), bounds

def prepare_crop(crop):
    """Upscale and normalize contrast; OCR still receives the entire model crop."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scale = max(1, 96 / gray.shape[0])
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
