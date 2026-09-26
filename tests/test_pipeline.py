import sqlite3
import numpy as np
import pytest
from plate_pipeline.geometry import iou, padded_crop
from plate_pipeline.metrics import edit_distance, match_boxes
from plate_pipeline.text import clean_text, valid_plate
from plate_pipeline.pipeline import PlatePipeline
from plate_pipeline.parking import ParkingLog
from plate_pipeline.ocr import OCR, ordered_text
from evaluate import summarize

def test_iou():
    assert iou([0,0,10,10],[0,0,10,10])==1
    assert iou([0,0,10,10],[20,20,30,30])==0
    assert iou([0,0,10,10],[5,0,15,10])==pytest.approx(1/3)
    assert iou([0,0,0,0],[0,0,0,0])==0

def test_padding_and_edges():
    image=np.arange(30*60*3,dtype=np.uint8).reshape(30,60,3)
    crop,box=padded_crop(image,[0,1,20,11],0.1)
    assert box==[0,0,22,12]
    np.testing.assert_array_equal(crop,image[:12,:22])
    crop[:]=0
    assert image[:12,:22].any()
    with pytest.raises(ValueError): padded_crop(image,[80,0,90,10])

def test_multiple_boxes_use_separate_pixels():
    image=np.zeros((30,60,3),dtype=np.uint8)
    image[:,:30]=42; image[:,30:]=200
    assert padded_crop(image,[5,5,20,20],0)[0].mean()==42
    assert padded_crop(image,[35,5,50,20],0)[0].mean()==200

def test_format_correction():
    assert clean_text('1O-A8-12I','az')=='10AB121'
    assert valid_plate('10AB121','az')
    assert clean_text('A8C-12O4','br')=='ABC1204'
    assert clean_text('A8C-12O4','generic')=='A8C12O4'
    assert not valid_plate(clean_text('uncertain text','az'),'az')
    assert clean_text('12','az')=='12'
    assert clean_text('NZF7823\n9CAO8','br')=='NZF7823'
    assert clean_text('ABC1234\nDEF5678','br')=='ABC1234DEF5678'

def test_one_to_one_matching():
    gt=[{'box':[0,0,10,10]}]
    preds=[{'box':[0,0,10,10],'confidence':0.9},{'box':[0,0,10,10],'confidence':0.8}]
    assert match_boxes(preds,gt)==[(0,0,1.0)]
    assert match_boxes([],gt)==[]
    assert match_boxes(preds,[])==[]

def test_cer():
    assert edit_distance('ABC1234','ABC134')==1
    assert edit_distance('ABC1234','')==7
    assert edit_distance('','XYZ')==3

def prediction(text='ABC1234'):
    return {'box':[0,0,10,10],'crop_box':[0,0,11,11],'confidence':0.9,
            'raw_text':text+'\n','text':text,'format_valid':True}

def test_metrics_penalize_misses_duplicates_and_bad_text():
    record={'image':'x.jpg','plates':[{'box':[0,0,10,10],'text':'ABC1234'}]}
    metrics,rows=summarize([record,record],[[prediction()],[]])
    assert metrics['mean_iou_all_gt']==0.5
    assert metrics['end_to_end_exact_match']==0.5
    assert metrics['end_to_end_cer']==0.5
    assert rows[1]['status']=='miss'
    metrics,rows=summarize([record],[[prediction(),prediction()]])
    assert metrics['precision_at_50']==0.5
    assert metrics['all_plates_image_accuracy']==0
    assert rows[-1]['status']=='false_positive'
    metrics,_=summarize([record],[[prediction('ABC123')]])
    assert metrics['end_to_end_cer']==pytest.approx(1/7)

def test_no_detections_never_calls_ocr():
    pipe=object.__new__(PlatePipeline)
    pipe.detect=lambda image: []
    assert pipe(np.zeros((10,10,3),dtype=np.uint8))==[]
    with pytest.raises(ValueError): pipe(None)

def test_log_deduplicates_and_filters_invalid(tmp_path):
    db=tmp_path/'log.db'; log=ParkingLog(db)
    p=prediction()
    log.record([p,p],'camera0')
    log.record([{**p,'text':'bad','format_valid':False}],'camera0')
    log.close()
    with sqlite3.connect(db) as conn:
        rows=conn.execute('SELECT timestamp,plate FROM sightings').fetchall()
    assert len(rows)==1 and rows[0][1]=='ABC1234' and '+00:00' in rows[0][0]

def test_ocr_keeps_raw_dealer_text_but_reads_large_characters():
    parts=[([[0,0],[40,0],[40,8],[0,8]],'STATE',0.8),
           ([[0,10],[100,10],[100,40],[0,40]],'ABC1234',0.9)]
    raw,confidence,number=ordered_text(parts)
    assert 'STATE' in raw and 'ABC1234' in raw
    assert number=='ABC1234'


def test_two_line_reading_order_and_country_marker():
    def part(x,y,text):
        return ([[x,y],[x+60,y],[x+60,y+20],[x,y+20]],text,.9)
    raw,_,number=ordered_text([part(80,40,'345'),part(50,5,'12'),part(5,40,'AB'),
                             ([[0,5],[10,5],[10,13],[0,13]],'AZ',.9)])
    assert 'AZ' in raw
    assert number=='12\nAB345'
    assert clean_text(number)=='12AB345'


def test_az_rules_do_not_invent_missing_characters():
    assert clean_text('AZ\n10-AB-123')=='10AB123'
    assert clean_text('77 RZ 1LL')=='77RZ144'
    assert clean_text('99TB4051')=='99TB405'
    assert clean_text('12AB34')=='12AB34'
    assert not valid_plate(clean_text('12AB34'))
    assert clean_text('12AB345\n67CD890')=='12AB34567CD890'


def test_unreadable_background_is_detection_only():
    gt={'image':'x.png','plates':[{'box':[0,0,10,10],'text':None}]}
    metrics,rows=summarize([gt],[[prediction()]])
    assert metrics['true_positives']==1
    assert metrics['readable_ground_truth_plates']==0
    assert rows[0]['status']=='detection_only'
    assert rows[0]['correct'] is None


def test_manifest_has_disjoint_identity_groups_and_24_test_images():
    import json
    from pathlib import Path
    records=json.loads((Path(__file__).parents[1]/'data/manifest.json').read_text())['images']
    test=[r for r in records if r['split']=='test']
    development=[r for r in records if r['split']=='development']
    assert len(test)==24 and len(records)==58
    assert not {r['group'] for r in test}&{r['group'] for r in development}
    assert not any(r['previously_seen'] for r in test)
