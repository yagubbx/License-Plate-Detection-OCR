"""Convert source pixel boxes to YOLO labels, then fine-tune on the train split."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import os
import cv2

os.environ.setdefault('YOLO_CONFIG_DIR', str(Path('.cache/ultralytics').resolve()))
os.environ.setdefault('MPLCONFIGDIR', str(Path('.cache/matplotlib').resolve()))

def prepare(manifest):
    records=json.loads(Path(manifest).read_text())['images']
    identities={}; hashes={}
    for r in records:
        for p in r['plates']:
            if p['text'] in identities and identities[p['text']] != r['split']:
                raise ValueError('Plate identity leaked between splits')
            identities[p['text']]=r['split']
        digest=hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()
        if digest in hashes and hashes[digest] != r['split']:
            raise ValueError('Image leaked between splits')
        hashes[digest]=r['split']
        image=cv2.imread(r['image']); h,w=image.shape[:2]
        if digest != r['sha256']:
            raise ValueError('Dataset image checksum mismatch')
        target=Path('data/yolo/images')/r['split']/Path(r['image']).name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(r['image'],target)
        labels=[]
        for p in r['plates']:
            x1,y1,x2,y2=p['box']
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError(f'Invalid source box: {r["id"]}')
            labels.append(f'0 {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}')
        label=Path('data/yolo/labels')/r['split']/(target.stem+'.txt')
        label.parent.mkdir(parents=True,exist_ok=True); label.write_text('\n'.join(labels)+'\n')
    Path('data/plates.yaml').write_text('path: '+Path('data/yolo').resolve().as_posix()+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: license_plate\n')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=25)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--serial-scan',action='store_true',help='Avoid Windows named pipes when scanning dataset labels')
    args=parser.parse_args()
    prepare('data/manifest.json')
    if args.prepare_only: return
    import torch
    from ultralytics import YOLO
    if args.serial_scan:
        # multiprocessing.ThreadPool creates a named pipe even though it uses threads.
        # A serial scan works in restricted Windows environments and changes no labels.
        import ultralytics.data.dataset as dataset_module
        class SerialPool:
            def __init__(self,*args,**kwargs): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def imap(self,func,iterable): return map(func,iterable)
        dataset_module.ThreadPool=SerialPool
    torch.set_num_threads(4)
    model=YOLO('yolov8n.pt')
    model.train(data='data/plates.yaml',epochs=args.epochs,imgsz=416,batch=8,device='cpu',workers=0,
                seed=42,deterministic=True,patience=12,project='runs',name='plate_train',exist_ok=True,
                degrees=8,translate=0.08,scale=0.25,fliplr=0,flipud=0,mosaic=0.5,close_mosaic=5,
                plots=True)
    Path('models').mkdir(exist_ok=True)
    run_dir=Path(model.trainer.save_dir)
    shutil.copy2(run_dir/'weights/best.pt','models/plate.pt')
    Path('reports/training').mkdir(parents=True,exist_ok=True)
    for name in ['args.yaml','results.csv','results.png','confusion_matrix.png']:
        p=run_dir/name
        if p.exists(): shutil.copy2(p,Path('reports/training')/name)

if __name__ == '__main__': main()
