"""Run both OCR engines on identical model detections from the held-out split."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import cv2
from plate_pipeline import PlatePipeline
from plate_pipeline.geometry import iou, padded_crop
from plate_pipeline.metrics import edit_distance, match_boxes
from plate_pipeline.ocr import OCR
from plate_pipeline.pipeline import draw_predictions
from plate_pipeline.text import normalize

def summarize(records, predictions):
    rows=[]; tp=fp=fn=exact=chars=errors=ocr_exact=ocr_errors=ocr_chars=0
    overlaps=[]; image_correct=0
    for r,preds in zip(records,predictions):
        truth=r['plates']; matches=match_boxes(preds,truth)
        mapped={gi:(pi,overlap) for pi,gi,overlap in matches}
        used={pi for pi,_,_ in matches}
        tp+=len(matches); fp+=len(preds)-len(matches); fn+=len(truth)-len(matches)
        all_correct=len(preds)==len(truth)==len(matches)
        # Mean IoU includes misses as zero and uses one-to-one positive-overlap assignment.
        weak={gi:ov for _,gi,ov in match_boxes(preds,truth,threshold=1e-9)}
        for gi,gt in enumerate(truth):
            overlaps.append(weak.get(gi,0.0))
            match=mapped.get(gi)
            pred=preds[match[0]] if match else None
            expected=normalize(gt['text']); actual=pred['text'] if pred else ''
            distance=edit_distance(expected,actual)
            correct=bool(pred is not None and actual==expected)
            exact+=correct; chars+=len(expected); errors+=distance; all_correct &= correct
            if pred:
                ocr_exact+=correct; ocr_errors+=distance; ocr_chars+=len(expected)
            rows.append({'image':r['image'],'ground_truth_box':json.dumps(gt['box']),
                         'ground_truth_text':expected,'detected_box':json.dumps(pred['box']) if pred else '',
                         'crop_box':json.dumps(pred['crop_box']) if pred else '',
                         'iou':match[1] if match else 0,'raw_text':pred['raw_text'] if pred else '',
                         'cleaned_text':actual,'format_valid':pred['format_valid'] if pred else False,
                         'correct':correct,'status':'correct' if correct else 'ocr_error' if pred else 'miss',
                         'character_errors':distance})
        for pi,pred in enumerate(preds):
            if pi not in used:
                rows.append({'image':r['image'],'ground_truth_box':'','ground_truth_text':'',
                             'detected_box':json.dumps(pred['box']),'crop_box':json.dumps(pred['crop_box']),
                             'iou':0,'raw_text':pred['raw_text'],'cleaned_text':pred['text'],
                             'format_valid':pred['format_valid'],'correct':False,'status':'false_positive',
                             'character_errors':''})
        image_correct+=all_correct
    total=sum(len(r['plates']) for r in records)
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/total if total else 0
    return {'images':len(records),'ground_truth_plates':total,'true_positives':tp,'false_positives':fp,
            'false_negatives':fn,'mean_iou_all_gt':sum(overlaps)/total if total else 0,
            'precision_at_50':precision,'recall_at_50':recall,
            'f1_at_50':2*precision*recall/(precision+recall) if precision+recall else 0,
            'end_to_end_exact_match':exact/total if total else 0,'exact_plates':exact,
            'end_to_end_cer':errors/chars if chars else 0,
            'ocr_exact_match_on_matched_detections':ocr_exact/tp if tp else None,
            'ocr_cer_on_matched_detections':ocr_errors/ocr_chars if ocr_chars else None,
            'all_plates_image_accuracy':image_correct/len(records) if records else 0},rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',default='models/plate.pt')
    parser.add_argument('--engines',nargs='+',default=['easyocr','tesseract'],choices=['easyocr','tesseract'])
    parser.add_argument('--tesseract-cmd')
    parser.add_argument('--manifest',default='data/manifest.json')
    parser.add_argument('--output',default='reports')
    args=parser.parse_args()
    import torch
    torch.set_num_threads(4)
    records=[r for r in json.loads(Path(args.manifest).read_text())['images'] if r['split']=='test']
    if len(records)<20: raise ValueError('Need at least 20 held-out test images')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    pipe=PlatePipeline(args.weights,args.engines[0],plate_format='br',tesseract_cmd=args.tesseract_cmd)
    images=[pipe.load_image(r['image']) for r in records]
    detections=[pipe.detect(image) for image in images]
    all_metrics={}; all_predictions={}
    for engine in args.engines:
        if engine != args.engines[0]: pipe.ocr=OCR(engine,tesseract_cmd=args.tesseract_cmd)
        predictions=[]; start=time.perf_counter()
        cropdir=out/engine/'crops'; cropdir.mkdir(parents=True,exist_ok=True)
        for record,image,boxes in zip(records,images,detections):
            pipe.plate_format=record['format']
            preds=[pipe.recognize(image,d) for d in boxes]
            predictions.append(preds)
            cv2.imwrite(str(out/engine/(record['id']+'.jpg')),draw_predictions(image,preds))
            for k,pred in enumerate(preds):
                crop,bounds=padded_crop(image,pred['box'],pipe.padding)
                assert bounds==pred['crop_box'] and crop.size
                cv2.imwrite(str(cropdir/(record['id']+f'_{k}.png')),crop)
            print(engine,record['id'],[p['text'] for p in preds],flush=True)
        metrics,rows=summarize(records,predictions)
        metrics['ocr_seconds']=time.perf_counter()-start
        all_metrics[engine]=metrics; all_predictions[engine]=predictions
        with (out/(engine+'.csv')).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
        (out/(engine+'_predictions.json')).write_text(json.dumps(predictions,indent=2),encoding='utf-8')
    metadata={'weights_sha256':hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
              'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
              'confidence':pipe.confidence,'padding':pipe.padding,'iou_threshold':0.5,'image_size':416,
              'metrics':all_metrics}
    (out/'metrics.json').write_text(json.dumps(metadata,indent=2))
    lines=['# Test results','',f'{len(records)} held-out images. All prediction boxes come from the trained model.',
           '', '| OCR | Mean IoU (all GT) | Precision@.5 | Recall@.5 | Exact match | CER | Image accuracy |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for engine,m in all_metrics.items():
        lines.append(f'| {engine} | {m["mean_iou_all_gt"]:.4f} | {m["precision_at_50"]:.2%} | {m["recall_at_50"]:.2%} | {m["end_to_end_exact_match"]:.2%} | {m["end_to_end_cer"]:.4f} | {m["all_plates_image_accuracy"]:.2%} |')
    lines+=['','Mean IoU uses confidence-ordered, one-to-one positive-overlap matching. Unmatched ground-truth plates contribute zero. Detection precision/recall and end-to-end accuracy require IoU >= 0.5.',
            '', 'CER is total Levenshtein distance divided by total ground-truth characters. Missed plates count as deleted strings. Extra detections count as false positives and fail image accuracy; they do not add characters to the per-ground-truth CER. OCR-only scores in metrics.json use matched model crops, never ground-truth crops.',
            '', 'Both engines receive identical detector boxes, padding and contrast preprocessing. The CSV files retain raw OCR output, cleaned text and every false positive.','',
            'See [EasyOCR rows](easyocr.csv), [Tesseract rows](tesseract.csv), and [failure analysis](failures.md).']
    (out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__': main()
