"""Run both OCR engines on identical model detections from the held-out split."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings
import cv2
from plate_pipeline import PlatePipeline
from plate_pipeline.geometry import padded_crop
from plate_pipeline.metrics import edit_distance, match_boxes
from plate_pipeline.ocr import OCR
from plate_pipeline.text import normalize

def summarize(records, predictions):
    rows=[]; tp=fp=fn=exact=chars=errors=ocr_exact=ocr_errors=ocr_chars=0
    overlaps=[]; image_correct=0; readable=matched_readable=primary_total=primary_exact=0
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
            expected=normalize(gt['text']) if gt['text'] is not None else None
            actual=pred['text'] if pred else ''
            distance=edit_distance(expected,actual) if expected is not None else 0
            correct=bool(pred is not None and actual==expected) if expected is not None else None
            if expected is not None:
                readable+=1; exact+=correct; chars+=len(expected); errors+=distance; all_correct &= correct
                if gt.get('primary',False):
                    primary_total+=1; primary_exact+=correct
            if pred and expected is not None:
                matched_readable+=1
                ocr_exact+=correct; ocr_errors+=distance; ocr_chars+=len(expected)
            rows.append({'image':r['image'],'ground_truth_box':json.dumps(gt['box']),
                         'ground_truth_text':expected or '(unreadable/partial)','primary':gt.get('primary',False),
                         'detected_box':json.dumps(pred['box']) if pred else '',
                         'crop_box':json.dumps(pred['crop_box']) if pred else '',
                         'iou':match[1] if match else 0,'raw_text':pred['raw_text'] if pred else '',
                         'cleaned_text':actual,'format_valid':pred['format_valid'] if pred else False,
                         'correct':correct,'status':('detection_only' if pred else 'miss') if expected is None else 'correct' if correct else 'ocr_error' if pred else 'miss',
                         'character_errors':distance})
        for pi,pred in enumerate(preds):
            if pi not in used:
                rows.append({'image':r['image'],'ground_truth_box':'','ground_truth_text':'','primary':False,
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
            'readable_ground_truth_plates':readable,
            'primary_plates':primary_total,'primary_exact_plates':primary_exact,
            'primary_plate_accuracy':primary_exact/primary_total if primary_total else None,
            'end_to_end_exact_match':exact/readable if readable else 0,'exact_plates':exact,
            'end_to_end_cer':errors/chars if chars else 0,
            'ocr_exact_match_on_matched_detections':ocr_exact/matched_readable if matched_readable else None,
            'ocr_cer_on_matched_detections':ocr_errors/ocr_chars if ocr_chars else None,
            'all_plates_image_accuracy':image_correct/len(records) if records else 0},rows

def write_report(out, records, predictions, metadata):
    """One report with results, raw OCR and up to three illustrated failures."""
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    engine=next(iter(predictions))
    _,rows=summarize(records,predictions[engine])
    split=metadata['split']
    label='held-out test' if split=='test' else split+' (not a held-out score)'
    lines=['# Plate recognition report','',f'{len(records)} {label} images. Boxes come from YOLOv8; labels are used only to check results.',
           '', '| OCR | Mean IoU | Precision | Recall | Exact match | CER | Whole-image accuracy |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in metadata['metrics'].items():
        lines.append(f'| {name} | {m["mean_iou_all_gt"]:.4f} | {m["precision_at_50"]:.1%} | {m["recall_at_50"]:.1%} | {m["exact_plates"]}/{m["readable_ground_truth_plates"]} ({m["end_to_end_exact_match"]:.1%}) | {m["end_to_end_cer"]:.4f} | {m["all_plates_image_accuracy"]:.1%} |')
    m=metadata['metrics'][engine]
    lines+=['',f'{engine}: the main plate is correct in **{m["primary_exact_plates"]}/{m["primary_plates"]}** images. The table above also counts background plates.',
            '', 'IoU checks box overlap. Missed boxes add zero to mean IoU. A correct detection needs IoU >= 0.5. Exact match means the full text is correct. CER is character edits / expected characters; missed text counts as deleted characters.',
            '', 'Unreadable/partial plates count for detection only. Plates below 25 x 8 pixels are outside the test scope. Extra detections lower precision and whole-image accuracy. Both OCR tools use the same model boxes.',
            '', '## Results', '', f'{engine} output. Box order: left, top, right, bottom. Raw text is shown before cleaning.',
            '', '| Image | Expected | Model box | Raw OCR | Clean text | Result |', '|---|---|---|---|---|---|']
    def cell(value):
        return str(value).replace('|','/').replace('\n',' / ').replace('\r','') or '-'
    for row in rows:
        box=', '.join(str(round(v)) for v in json.loads(row['detected_box'])) if row['detected_box'] else '-'
        lines.append('| '+' | '.join(cell(v) for v in [Path(row['image']).name,row['ground_truth_text'],box,row['raw_text'],row['cleaned_text'],row['status']])+' |')
    lines+=['', '## Failure examples', '', 'These crops use model boxes, not hand-drawn boxes. Causes below describe likely OCR problems.']
    failures=[r for r in rows if r['status']=='ocr_error']
    failures+=[r for r in rows if r['status'] in ('miss','false_positive')]
    chosen=[failures[i] for i in sorted({0,len(failures)//2,len(failures)-1})] if failures else []
    # Prefer three OCR errors when available so the hand-off can be inspected.
    ocr_errors=[r for r in rows if r['status']=='ocr_error']
    if len(ocr_errors)>=3:
        chosen=[ocr_errors[i] for i in (0,len(ocr_errors)//2,len(ocr_errors)-1)]
    for index,row in enumerate(chosen,1):
        image=PlatePipeline.load_image(row['image'])
        if row['crop_box']:
            x1,y1,x2,y2=json.loads(row['crop_box']); image=image[y1:y2,x1:x2]
        pictures=out/'images'; pictures.mkdir(exist_ok=True)
        path=pictures/f'failure_{index}.png'
        if not cv2.imwrite(str(path),image): raise RuntimeError(f'Cannot save {path}')
        if row['status']=='miss': reason='The detector missed this plate. Small size, blur or occlusion can cause this.'
        elif row['status']=='false_positive': reason='The detector marked a region that is not an annotated plate.'
        elif len(row['cleaned_text'])!=len(row['ground_truth_text']): reason='The box covers the plate, but OCR drops or adds characters. Repeated narrow characters can merge. The format rule cannot restore missing text.'
        elif 'Z' in row['raw_text'] and any(a=='4' and b=='2' for a,b in zip(row['ground_truth_text'],row['cleaned_text'])): reason='OCR reads an open 4 as Z; the digit rule changes Z to 2. The wrong digit still passes the format check.'
        else: reason='The box covers the plate, but OCR confuses similar character shapes. A wrong digit can still pass the format check.'
        lines+=['',f'### {Path(row["image"]).name}', '',f'![Model crop](images/{path.name})', '',
                f'Expected `{row["ground_truth_text"]}`; read `{row["cleaned_text"] or "empty"}`. {reason}']
    if not chosen: lines+=['','No failures in this run.']
    if metadata['manifest_sha256']=='a33a76aab546218ccca0572057d9342cd9ef27343b100737c13a3d3e3cc3d378':
        lines+=['','## Other checks (saved validation)',
                '', '58 supplied photos: 34 development, 24 test. The same plate stays in one split. No weights were trained on these photos. Across both splits, 53/58 main plates were correct; this is not the held-out score.',
                '', 'Multiple vehicles: car_5 (2 crops), car_23 (2), car_25 (3). All 7 model crops had 8% padding, stayed within image bounds and matched the source pixels. Some background text was unreadable.',
                '', 'Two-line OCR: 12/AB345, 90/XY678 and 77/RZ144 all passed on synthetic crops. Real two-line vehicle photos were not available.',
                '', 'Tesseract is optional. The saved comparison used version 5.4.0, extracted locally because its installer could not run. EasyOCR needs only pip installation.']
    lines+=['', 'Settings: image size 960; confidence 0.25; crop padding 8%.',
            '',f'Model SHA-256: `{metadata["weights_sha256"]}`',
            '',f'Labels SHA-256: `{metadata["manifest_sha256"]}`']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',default='models/plate.pt')
    parser.add_argument('--engines',nargs='+',default=['easyocr'],choices=['easyocr','tesseract'])
    parser.add_argument('--tesseract-cmd')
    parser.add_argument('--manifest',default='data/manifest.json')
    parser.add_argument('--output',default='reports')
    parser.add_argument('--split',choices=['test','development','all'],default='test')
    args=parser.parse_args()
    warnings.filterwarnings('ignore',message='.*pin_memory.*no accelerator.*',category=UserWarning)
    warnings.filterwarnings('ignore',message='torch.quantize_per_tensor, torch.quantize_per_channel.*',category=UserWarning)
    import torch
    torch.set_num_threads(4)
    records=[r for r in json.loads(Path(args.manifest).read_text())['images'] if args.split=='all' or r['split']==args.split]
    if args.split=='test' and len(records)<20: raise ValueError('Need at least 20 labeled held-out test images.')
    if not records: raise ValueError('The requested split is empty.')
    for record in records:
        if record.get('plates') is None: raise ValueError(f'Missing ground truth: {record["image"]}')
        if hashlib.sha256(Path(record['image']).read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError(f'Image changed since annotation: {record["image"]}')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    # Check the optional system dependency before starting a long EasyOCR run.
    tesseract=OCR('tesseract',tesseract_cmd=args.tesseract_cmd) if 'tesseract' in args.engines else None
    pipe=PlatePipeline(args.weights,args.engines[0],plate_format='az',tesseract_cmd=args.tesseract_cmd)
    images=[pipe.load_image(r['image']) for r in records]
    detections=[[d for d in pipe.detect(image) if d['box'][2]-d['box'][0]>=25 and d['box'][3]-d['box'][1]>=8] for image in images]
    all_metrics={}; all_predictions={}
    for engine in args.engines:
        if engine != args.engines[0]: pipe.ocr=tesseract if engine=='tesseract' else OCR(engine)
        predictions=[]; start=time.perf_counter()
        for record,image,boxes in zip(records,images,detections):
            pipe.plate_format=record['format']
            preds=[pipe.recognize(image,d) for d in boxes]
            predictions.append(preds)
            for k,pred in enumerate(preds):
                crop,bounds=padded_crop(image,pred['box'],pipe.padding)
                assert bounds==pred['crop_box'] and crop.size
            print(f'{engine:10} {record["id"]:8} '+(' | '.join(p['text'] or '(unreadable)' for p in preds) or 'No plate detected'),flush=True)
        metrics,rows=summarize(records,predictions)
        metrics['ocr_seconds']=time.perf_counter()-start
        all_metrics[engine]=metrics; all_predictions[engine]=predictions
    metadata={'weights_sha256':hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
              'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
              'split':args.split,'confidence':pipe.confidence,'padding':pipe.padding,'iou_threshold':0.5,'image_size':pipe.image_size,
              'metrics':all_metrics}
    write_report(out,records,all_predictions,metadata)
    for engine,m in all_metrics.items():
        print(f'\n{engine.upper()} RESULT\n  Mean IoU   : {m["mean_iou_all_gt"]:.4f}\n  Exact match: {m["exact_plates"]}/{m["readable_ground_truth_plates"]} ({m["end_to_end_exact_match"]:.1%})\n  CER        : {m["end_to_end_cer"]:.4f}')
    print(f'\nReports saved to: {out.resolve()}')

if __name__=='__main__': main()
