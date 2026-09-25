# Multiple-vehicle crop checks

These three retained images contain neighbouring vehicles as well as the labeled foreground car. They are part of the 25-image evaluation set, not additional dataset images.

| Image | Detected text | Annotated result | Model crop |
|---|---|---|---|
| car_19 | JRK5336 | [Result](easyocr/car_19.jpg) | [Crop](easyocr/crops/car_19_0.png) |
| car_20 | PYB6477 | [Result](easyocr/car_20.jpg) | [Crop](easyocr/crops/car_20_0.png) |
| car_21 | AYO9034 | [Result](easyocr/car_21.jpg) | [Crop](easyocr/crops/car_21_0.png) |

Each foreground plate was correctly read. The evaluator checks every saved model crop for nonempty pixels and the expected padded/clipped bounds. Padding is 8% on each side of the detector box. Annotation coordinates are used only for IoU matching, never for cropping during inference.

The supplied car_2 also contains two labeled plates. The foreground plate was found, but the smaller background plate was missed and counts as a false negative. Supporting multiple returned detections does not guarantee that every plate is found.

Unit tests also cover separate crops from multiple boxes, boundary clipping, no detections and duplicate matching. The video loop supports webcam input and video files; processing speed depends on the machine. No physical webcam was tested.

The updated video command was checked on a temporary nine-frame clip made from car_19, car_22 and car_23. All nine output frames were written at 2.64 processing FPS on the test CPU. SQLite recorded JRK5336, JRV1942 and PJH0957 once each, suppressing consecutive repeats. The temporary clip is not an additional dataset image or a shipped asset.
