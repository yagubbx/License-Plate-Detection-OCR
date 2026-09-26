import argparse
import json
import time
import warnings
import sys
from contextlib import redirect_stdout
from pathlib import Path
import cv2
from plate_pipeline import PlatePipeline
from plate_pipeline.pipeline import draw_predictions
from plate_pipeline.parking import ParkingLog

def print_result(source, results, output):
    line = '=' * 62
    print(f'\n{line}\n  LICENSE PLATE RECOGNITION\n{line}')
    print(f'  Image       : {Path(source).name}')
    print(f'  Plates found: {len(results)}')
    for index, result in enumerate(results, 1):
        valid = result['format_valid']
        status = 'Matches format' if valid is True else 'Check reading' if valid is False else 'Not checked'
        print(f'\n  PLATE {index}    {result.get("display_text",result["text"]) or "Unreadable"}')
        print('  ' + '-' * 58)
        print(f'  Detection   : {result["confidence"]:.1%}')
        if result.get('ocr_confidence') is not None:
            print(f'  OCR score   : {result["ocr_confidence"]:.1%}')
        print(f'  Format      : {status}')
        print(f'  Box (xyxy)  : {", ".join(str(round(v)) for v in result["box"])}')
        print(f'  Raw OCR     : {" / ".join(result["raw_text"].split()) or "(empty)"}')
    if not results:
        print('\n  No plate detected. Try a clearer or closer image.')
    print(f'\n  Saved image : {Path(output)}')
    print('  Scores and format checks do not guarantee a correct reading.')
    print(line + '\n')

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
    p.add_argument('--json',action='store_true',help='Print machine-readable JSON instead of the result summary')
    args=p.parse_args()
    # These CPU/deprecation notices are unrelated to the result; keep other warnings visible.
    warnings.filterwarnings('ignore', message='.*pin_memory.*no accelerator.*', category=UserWarning)
    warnings.filterwarnings('ignore', message='torch.quantize_per_tensor, torch.quantize_per_channel.*', category=UserWarning)
    import torch
    torch.set_num_threads(4)
    with redirect_stdout(sys.stderr):
        pipe=PlatePipeline(args.weights,args.ocr,args.format,tesseract_cmd=args.tesseract_cmd)
    log=ParkingLog(args.db) if args.db else None
    try:
        if args.image:
            image=pipe.load_image(args.image); results=pipe.predict(image)
            Path(args.output).parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(args.output,draw_predictions(image,results)):
                raise RuntimeError('Could not write output image')
            if log: log.record(results,args.image)
            if args.json:
                print(json.dumps(results,indent=2))
            else:
                print_result(args.image,results,args.output)
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
            fps=count/max(1e-9,time.perf_counter()-start)
            if args.json:
                print(json.dumps({'frames':count,'processing_fps':fps}))
            else:
                print(f'\nVIDEO COMPLETE\n  Frames processed : {count}\n  Processing speed : {fps:.2f} FPS')
                if writer: print(f'  Saved video      : {args.output}')
        finally:
            cap.release()
            if writer: writer.release()
            if not args.headless: cv2.destroyAllWindows()
    finally:
        if log: log.close()

if __name__ == '__main__': main()
