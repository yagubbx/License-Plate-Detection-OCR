import argparse
import json
import time
from pathlib import Path
import cv2
from plate_pipeline import PlatePipeline
from plate_pipeline.pipeline import draw_predictions
from plate_pipeline.parking import ParkingLog

def main():
    p=argparse.ArgumentParser(description='Detect and read every license plate in an image or video.')
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--image')
    group.add_argument('--video',help='Video path or camera index, e.g. 0')
    p.add_argument('--weights',default='models/plate.pt')
    p.add_argument('--ocr',choices=['easyocr','tesseract'],default='easyocr')
    p.add_argument('--format',choices=['az','br','generic'],default='az')
    p.add_argument('--tesseract-cmd')
    p.add_argument('--output',default='prediction.jpg')
    p.add_argument('--db',help='Optional SQLite parking log path')
    p.add_argument('--headless',action='store_true',help='Process a video without opening a window')
    args=p.parse_args()
    import torch
    torch.set_num_threads(4)
    pipe=PlatePipeline(args.weights,args.ocr,args.format,tesseract_cmd=args.tesseract_cmd)
    log=ParkingLog(args.db) if args.db else None
    try:
        if args.image:
            image=pipe.load_image(args.image); results=pipe.predict(image)
            print(json.dumps(results,indent=2))
            Path(args.output).parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(args.output,draw_predictions(image,results)):
                raise RuntimeError('Could not write output image')
            if log: log.record(results,args.image)
            return
        source=int(args.video) if args.video.isdigit() else args.video
        cap=cv2.VideoCapture(source)
        if not cap.isOpened(): raise RuntimeError(f'Cannot open video source: {source}')
        writer=None; count=0; start=time.perf_counter()
        try:
            while True:
                ok,frame=cap.read()
                if not ok: break
                results=pipe.predict(frame)
                if log: log.record(results,args.video)
                canvas=draw_predictions(frame,results); count+=1
                fps=count/(time.perf_counter()-start)
                cv2.putText(canvas,f'{fps:.1f} processing FPS',(12,24),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
                if args.output != 'prediction.jpg':
                    if writer is None:
                        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
                        rate=cap.get(cv2.CAP_PROP_FPS)
                        writer=cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc(*'mp4v'),rate if 0 < rate < 240 else 25,(canvas.shape[1],canvas.shape[0]))
                        if not writer.isOpened(): raise RuntimeError('Could not open output video')
                    writer.write(canvas)
                if not args.headless:
                    cv2.imshow('License plates',canvas)
                    if cv2.waitKey(1)&0xff == ord('q'): break
            print(json.dumps({'frames':count,'processing_fps':count/max(1e-9,time.perf_counter()-start)}))
        finally:
            cap.release()
            if writer: writer.release()
            if not args.headless: cv2.destroyAllWindows()
    finally:
        if log: log.close()

if __name__ == '__main__': main()
