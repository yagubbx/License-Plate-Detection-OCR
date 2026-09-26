# Evaluation results

24 held-out test images. All prediction boxes come from YOLOv8.

| OCR | Mean IoU (all GT) | Precision@.5 | Recall@.5 | Exact match | CER | Image accuracy |
|---|---:|---:|---:|---:|---:|---:|
| easyocr | 0.7606 | 96.43% | 90.00% | 76.00% | 0.0914 | 62.50% |
| tesseract | 0.7606 | 96.43% | 90.00% | 48.00% | 0.3600 | 41.67% |

Mean IoU uses confidence-ordered, one-to-one positive-overlap matching. Unmatched ground-truth plates contribute zero. Detection precision/recall and end-to-end accuracy require IoU >= 0.5.

CER is total Levenshtein distance divided by total ground-truth characters. Missed plates count as deleted strings. Extra detections count as false positives and fail image accuracy; they do not add characters to the per-ground-truth CER. OCR-only scores in metrics.json use matched model crops, never ground-truth crops.

Unreadable or partial background strings are marked null in the manifest: they count in detection metrics, but not OCR denominators. Plates smaller than 25x8 pixels are outside the annotation scope; predictions below that size are excluded only during evaluation.

Each engine receives identical detector boxes and padded source pixels. The CSV files retain raw OCR output, cleaned text and every false positive.

easyocr: primary vehicle plate correct in **19/24** images. Background detections are included in the stricter table above.

tesseract: primary vehicle plate correct in **12/24** images. Background detections are included in the stricter table above.

## Per-image results

| Image | GT | Detected box | OCR | Result |
|---|---|---|---|---|
| car_3.png | 90OT012 | 294, 569, 495, 623 | 90OT012 | correct |
| car_9.png | 20CT700 | 207, 747, 343, 783 | 20CT700 | correct |
| car_12.png | 55CE155 | 136, 640, 242, 688 | 55CE155 | correct |
| car_12.png |  | 49, 550, 284, 645 | - | false_positive |
| car_15.png | 77AJ460 | 397, 545, 555, 598 | 77AJ260 | ocr_error |
| car_17.png | 90PU029 | 389, 673, 560, 727 | 90PU029 | correct |
| car_18.png | 90UN023 | 468, 848, 598, 897 | 90UN023 | correct |
| car_22.png | 90ZC025 | 211, 573, 345, 615 | 90ZC025 | correct |
| car_25.png | 99FR770 | 366, 798, 565, 855 | 99FR770 | correct |
| car_25.png | (unreadable/partial) | 0, 443, 33, 459 | Z65 | detection_only |
| car_25.png | (unreadable/partial) | 666, 558, 736, 582 | 99G03 | detection_only |
| car_27.png | 99RT100 | 128, 769, 235, 812 | 99RT100 | correct |
| car_27.png | (unreadable/partial) | - | - | miss |
| car_31.png | 42CL107 | 304, 730, 483, 773 | 42CL107 | correct |
| car_31.png | (unreadable/partial) | 673, 666, 735, 687 | - | detection_only |
| car_35.png | 99FV543 | 264, 685, 488, 744 | 99FV543 | correct |
| car_35.png | (unreadable/partial) | - | - | miss |
| car_36.png | 77ES425 | 306, 693, 471, 737 | 77ES225 | ocr_error |
| car_37.png | 11LX111 | 236, 743, 490, 805 | EMLX | ocr_error |
| car_38.png | 90XC832 | 513, 801, 630, 842 | 90XC832 | correct |
| car_39.png | 77ES425 | 368, 682, 538, 727 | 77ES423 | ocr_error |
| car_40.png | 99MT073 | 428, 757, 622, 817 | 99MT073 | correct |
| car_41.png | 10ON010 | 239, 225, 436, 268 | 10ON010 | correct |
| car_42.png | 01AA001 | 192, 681, 299, 715 | 01AA001 | correct |
| car_43.png | 99DU911 | 185, 648, 361, 704 | 99DU911 | correct |
| car_43.png | 10FY911 | - | - | miss |
| car_49.png | 77SG387 | 68, 764, 184, 814 | 77SG387 | correct |
| car_51.png | 77SX585 | 361, 524, 536, 588 | 77SX585 | correct |
| car_52.png | 77KG155 | 303, 765, 469, 813 | 77KG155 | correct |
| car_53.png | 10UD527 | 477, 718, 601, 774 | 70UD527 | ocr_error |
| car_56.png | 77YH502 | 420, 571, 520, 601 | 77YH502 | correct |
