import math
import cv2
import numpy as np

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


def straighten_plate(crop):
    """Find the plate border inside a detector crop; leave it alone if uncertain."""
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY) if crop.ndim==3 else crop
    gray=cv2.resize(gray,None,fx=max(1,100/gray.shape[0]),fy=max(1,100/gray.shape[0]),interpolation=cv2.INTER_CUBIC)
    h,w=gray.shape
    edges=cv2.Canny(gray,50,150)
    masks=[edges,cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)),
           cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]]
    candidates=[]
    for mask in masks:
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area=cv2.contourArea(contour)
            if not .35*w*h < area < .96*w*h: continue
            quad=cv2.approxPolyDP(contour,.035*cv2.arcLength(contour,True),True)
            if len(quad)!=4 or not cv2.isContourConvex(quad): continue
            pts=quad.reshape(4,2).astype(np.float32)
            order=np.argsort(pts[:,0]); left=pts[order[:2]]; right=pts[order[2:]]
            tl,bl=left[np.argsort(left[:,1])]; tr,br=right[np.argsort(right[:,1])]
            width=max(np.linalg.norm(tr-tl),np.linalg.norm(br-bl))
            height=max(np.linalg.norm(bl-tl),np.linalg.norm(br-tr))
            if not 1.1 < width/max(1,height) < 7.5: continue
            candidates.append((area,np.array([tl,tr,br,bl]),int(width),int(height)))
    if not candidates:
        # A broken border may not form a quadrilateral. Long near-horizontal
        # edges still give a useful rotation, without inventing plate corners.
        lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=30,minLineLength=int(w*.45),maxLineGap=10)
        angles=[]
        for x1,y1,x2,y2 in lines.reshape(-1,4) if lines is not None else []:
            angle=np.degrees(np.arctan2(y2-y1,x2-x1))
            if abs(angle)<30: angles.append(angle)
        angle=float(np.median(angles)) if angles else 0
        if abs(angle)<2: return None
        matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
        nw=int(h*abs(matrix[0,1])+w*abs(matrix[0,0]))
        nh=int(h*abs(matrix[0,0])+w*abs(matrix[0,1]))
        matrix[:,2]+=[(nw-w)/2,(nh-h)/2]
        return cv2.warpAffine(gray,matrix,(nw,nh),borderMode=cv2.BORDER_REPLICATE)
    _,points,width,height=max(candidates,key=lambda p:p[0])
    target=np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
    matrix=cv2.getPerspectiveTransform(points,target)
    return cv2.warpPerspective(gray,matrix,(width,height),borderMode=cv2.BORDER_REPLICATE)
