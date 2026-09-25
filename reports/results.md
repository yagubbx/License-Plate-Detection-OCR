# Test results

30 held-out images. All prediction boxes come from the trained model.

| OCR | Mean IoU (all GT) | Precision@.5 | Recall@.5 | Exact match | CER | Image accuracy |
|---|---:|---:|---:|---:|---:|---:|
| easyocr | 0.8457 | 100.00% | 100.00% | 76.67% | 0.0714 | 76.67% |
| tesseract | 0.8457 | 100.00% | 100.00% | 3.33% | 0.8476 | 3.33% |

Mean IoU uses confidence-ordered, one-to-one positive-overlap matching. Unmatched ground-truth plates contribute zero. Detection precision/recall and end-to-end accuracy require IoU >= 0.5.

CER is total Levenshtein distance divided by total ground-truth characters. Missed plates count as deleted strings. Extra detections count as false positives and fail image accuracy; they do not add characters to the per-ground-truth CER. OCR-only scores in metrics.json use matched model crops, never ground-truth crops.

Both engines receive identical detector boxes, padding and contrast preprocessing. The CSV files retain raw OCR output, cleaned text and every false positive.

See [EasyOCR rows](easyocr.csv), [Tesseract rows](tesseract.csv), and [failure analysis](failures.md).
