# Multiple vehicles and difficult inputs

## Three real scenes

These test images contain a foreground vehicle and neighbouring vehicles. The source benchmark labels the foreground plate. The model detected it in each scene. Every saved crop was checked to equal the padded, clipped model box; none uses an annotation as an inference crop.

| Image | Detected text | Full prediction | Model crop |
|---|---|---|---|
| JRK5336 | JRK5336 | [Image](cases/JRK5336.jpg) | [Crop](cases/JRK5336_crop_0.png) |
| PYB6477 | PYB6477 | [Image](cases/PYB6477.jpg) | [Crop](cases/PYB6477_crop_0.png) |
| AYO9034 | AYO9034 | [Image](cases/AYO9034.jpg) | [Crop](cases/AYO9034_crop_0.png) |

## Multiple readable plates

Three side-by-side composites check the multi-detection loop more directly. They use held-out source images but are **synthetic checks**, not extra independent test images.

| Composite | Plates detected | Output |
|---|---:|---|
| [1](cases/composite_1.jpg) | 2/2 | JRV1942, PJH0957 |
| [2](cases/composite_2.jpg) | 2/2 | PJV9741, PJY5472 |
| [3](cases/composite_3.jpg) | 1/2 | DZK6717 |

The third composite fails: PJB7392 is missed and OZK6717 is read as DZK6717. Resizing two scenes into one reduces plate size. The code handles every returned box, but this small model does not detect every plate in every scene.

## Controlled stress checks

| Input | Detections | OCR result | Interpretation |
|---|---:|---|---|
| [Blur](cases/blur.jpg) | 1 | ET | Found the region; text unreadable and format invalid |
| [20-degree angle](cases/angle.jpg) | 1 | JRV1942 | Correct reading despite rotation |
| [Partial occlusion](cases/occlusion.jpg) | 0 | empty list | Detector misses the masked plate; no invented OCR result |
| [No plate](cases/no_plate.jpg) | 0 | empty list | No crash and no false plate on this uniform input |

These transformations are applied to JRV1942 and are excluded from headline metrics. Ground-truth coordinates are used only to place the synthetic occlusion, never to supply a detection or OCR crop. The uniform negative is a smoke test, not a representative negative-image benchmark.

## Video and parking log

The actual video command processed all 9 frames of [this generated test clip](cases/smoke_input.mp4) and wrote [the annotated output](cases/smoke_output.mp4). Processing speed on the test CPU was **2.37 FPS**. This confirms the video path; it is not evidence of full-frame-rate webcam performance. A physical webcam was not tested.

The same run wrote three unique UTC-timestamped entries: JRV1942, PJH0957 and JRK5336. Consecutive repeats were suppressed. [Recorded log rows](cases/parking_log.csv) show the actual database output.

Reproduce the checks with `python check_cases.py`. All case predictions are stored in [checks.json](cases/checks.json).
