# Start

Use **Python 3.11** and a PowerShell terminal in VS Code.

## 1. Get the project

If you already extracted the ZIP, open the folder with `run.py` and skip these commands.

```powershell
git clone https://github.com/yagubbx/License-Plate-Detection-OCR.git
cd License-Plate-Detection-OCR
```

This compact version is in the ZIP only until it is uploaded to GitHub.

## 2. Install once

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Read a plate

```powershell
.\.venv\Scripts\python.exe run.py --image data/images/car_1.png
```

Change `car_1.png` to another image name. The result appears in the terminal and in `prediction.jpg`. The first run downloads OCR models. Later runs use the saved models.

## 4. Check accuracy

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

Tests the 24 test images and updates `reports/report.md`.

## 5. Check the code (optional)

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Next time, start at step 3. You do not need to install again. Tesseract is not needed for these commands. Use `run.py --help` with the same Python command to see extra options.
