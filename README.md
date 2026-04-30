# AI Image Detector

An application that analyzes images and determines whether they are AI-generated or real photographs. Uses three independent detection layers and combines their results into a final verdict.


---

## How It Works

Every image is run through three detection layers. Each layer casts a vote on whether the image is AI or real. The votes are combined into a final verdict — the CNN layer counts double when it is confident since it is the strongest signal.

### Layer 1 — Metadata Analysis
Checks the image's EXIF data for signs that it was not taken by a real camera. Real photos from phones and cameras contain embedded metadata including the camera model, capture time, and software used. AI-generated images typically have no EXIF data or are missing key fields.

### Layer 2 — Pixel Analysis
Examines low-level pixel patterns that differ between real photos and AI-generated images. Checks five signals:
- **Noise** — AI images tend to have unnaturally smooth or uniform noise
- **Frequency domain** — AI generators leave characteristic patterns in the frequency spectrum
- **Edges** — AI images often have unusually low or artificially consistent edge density
- **Texture** — AI images tend to have overly uniform texture across regions
- **Color** — AI images often have lower color variance and entropy than real photos

### Layer 3 — CNN (EfficientNet-B0)
A deep learning model trained on AI-generated vs real images using two-phase fine-tuning. Uses test-time augmentation (TTA) — running multiple crops and flips of the image through the model and averaging the results — for more reliable predictions.

---


## Requirements

- Python 3.10 or higher
- The trained model file (see Installation below)

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/ai-image-detector.git
cd ai-image-detector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the trained model**

Download the trained model weights from the link below and place the file at `layers/cnn_ai_detector.pth` inside the project folder. Create the `layers/` folder if it does not exist.

**[Download cnn_ai_detector.pth](https://drive.google.com/file/d/1Wo4MOpKJW11RD-Mnjo3irKBTO_Yva7fC/view?usp=drive_link)**


---

## Running the Application

```bash
python main.py
```

The GUI will open. The CNN model loads in the background on startup — wait for the status indicator in the top right to show -Ready- before analyzing images.

For Analyizing Image
- Drag and drop any image file onto the drop zone
- Or click Browse File to open a file picker

Supported formats: `.jpg` `.jpeg` `.png` `.webp` `.bmp` `.tiff` `.jfif`

---

## Reading the Results

The results panel shows each layer's findings individually, then a final combined verdict at the bottom.

| Indicator | Meaning |
|-----------|---------|
| Green | Likely real photo |
| Red   | Likely AI-generated |
| Yellow| Low confidence — result uncertain |

The confidence percentage and vote breakdown (e.g. `AI 3 / Real 1`) are shown alongside the verdict so you can see how strongly the layers agreed.

---

## Notes

- The CNN model loads once at startup. Subsequent image analyses reuse the loaded model so there is no reload delay after the first image.
- On CPU (no GPU), each image analysis takes a few seconds due to test-time augmentation running multiple forward passes. On GPU it is near-instant.
- The model was trained on approximately 10,000 Images, 5000 AI-generated and 5,000 real images. It performs best on images from common AI generators. Accuracy may vary on images from newer or less common generators.

---

## License

MIT
