"""Save multiple-vehicle and difficult-input checks separately from the test metrics."""
import json
from pathlib import Path
import cv2
import numpy as np
from plate_pipeline import PlatePipeline
from plate_pipeline.geometry import padded_crop
from plate_pipeline.pipeline import draw_predictions
from plate_pipeline.metrics import match_boxes

def main():
    import torch
    torch.set_num_threads(4)
    pipe=PlatePipeline(plate_format='br')
    records={r['id']:r for r in json.loads(Path('data/manifest.json').read_text())['images'] if r['split']=='test'}
    out=Path('reports/cases'); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    # Real scenes with a foreground car and neighbouring vehicles, not composites.
    for name in ['JRK5336','PYB6477','AYO9034']:
        r=records[name]; image=pipe.load_image(r['image']); preds=pipe.predict(image)
        cv2.imwrite(str(out/(name+'.jpg')),draw_predictions(image,preds))
        crops=[]
        for i,p in enumerate(preds):
            crop,bounds=padded_crop(image,p['box'])
            assert bounds==p['crop_box'] and crop.size
            filename=f'{name}_crop_{i}.png'; cv2.imwrite(str(out/filename),crop); crops.append(filename)
        rows.append({'case':name,'type':'real_multiple_vehicle_scene','predictions':preds,'crops':crops})
    # Composites exercise two readable foreground plates, where the source benchmark
    # otherwise labels just one plate per scene. These are NOT additional test images.
    for index,pair in enumerate([('JRV1942','PJH0957'),('PJV9741','PJY5472'),('PJB7392','OZK6717')],1):
        frames=[]
        for name in pair:
            im=pipe.load_image(records[name]['image'])
            frames.append(cv2.resize(im,(480,480)))
        image=np.concatenate(frames,axis=1)
        preds=pipe.predict(image); name=f'composite_{index}'
        cv2.imwrite(str(out/(name+'_input.jpg')),image)
        cv2.imwrite(str(out/(name+'.jpg')),draw_predictions(image,preds))
        crops=[]
        for i,p in enumerate(preds):
            crop,_=padded_crop(image,p['box']); filename=f'{name}_crop_{i}.png'
            cv2.imwrite(str(out/filename),crop); crops.append(filename)
        rows.append({'case':name,'type':'synthetic_two_vehicle_seam_check','sources':pair,'predictions':preds,'crops':crops})
    r=records['JRV1942']; original=pipe.load_image(r['image']); h,w=original.shape[:2]
    blur=cv2.GaussianBlur(original,(31,31),9)
    rotation=cv2.getRotationMatrix2D((w/2,h/2),20,1)
    angled=cv2.warpAffine(original,rotation,(w,h),borderMode=cv2.BORDER_REPLICATE)
    hidden=original.copy(); x1,y1,x2,y2=r['plates'][0]['box']
    cv2.rectangle(hidden,(int(x1+(x2-x1)*0.4),y1),(int(x1+(x2-x1)*0.65),y2),(35,35,35),-1)
    for name,image in [('blur',blur),('angle',angled),('occlusion',hidden),('no_plate',np.full_like(original,150))]:
        preds=pipe.predict(image)
        cv2.imwrite(str(out/(name+'_input.jpg')),image)
        cv2.imwrite(str(out/(name+'.jpg')),draw_predictions(image,preds))
        rows.append({'case':name,'type':'synthetic_stress_check','source':'JRV1942',
                     'expected_text':r['plates'][0]['text'] if name!='no_plate' else '', 'predictions':preds})
    (out/'checks.json').write_text(json.dumps(rows,indent=2))
    # A short generated clip tests the actual VideoCapture -> pipeline -> VideoWriter path.
    video=cv2.VideoWriter(str(out/'smoke_input.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),5,(640,480))
    if not video.isOpened(): raise RuntimeError('Video writer failed')
    for name in ['JRV1942','PJH0957','JRK5336']:
        frame=cv2.resize(pipe.load_image(records[name]['image']),(640,480))
        for _ in range(3): video.write(frame)
    video.release()
    print(json.dumps([{'case':r['case'],'detections':len(r['predictions'])} for r in rows],indent=2))

if __name__=='__main__': main()
