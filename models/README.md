# Detector

`plate.pt` is the pretrained YOLOv8n license-plate detector published by [Koushim](https://huggingface.co/Koushim/yolov8-license-plate-detection). It has one class, `license_plate`. A configured pretrained detector meets the assignment; training another model is not needed to run this project.

- Source revision: `9aaa5cd490abe0c165882ba87f4f62658ab54d01`
- Source file: `best.pt`, renamed to `plate.pt`
- SHA-256: `2d95861825bb4184404344c9cf809f40fd31dba785fe54e8ba5b9a3583789822`
- Upstream model card declares MIT. The application code is AGPL-3.0, matching its Ultralytics dependency.
- Inference: image size 960, confidence 0.25, NMS IoU 0.5, CPU.

The previous Brazilian detector missed Azerbaijani plates. On 12 previously used development images, this replacement found all 13 readable annotated plates at IoU >= 0.5. Selection and OCR tuning used development images only. The 24 test images were reserved before OCR tuning; no weights were trained on this collection.

The publisher does not identify its training images. Overlap with upstream pretraining cannot be ruled out; “held out” refers to this project's development process, not a verified independent benchmark of the upstream weights.
