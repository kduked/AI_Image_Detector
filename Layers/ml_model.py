"""
Layer 3: CNN Detection
Loads a trained EfficientNet-B0 and predicts whether an image is AI-generated or real.
Trained by train_model.py — CLASS_TO_IDX must match between both files.
"""


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from pathlib import Path


# ── Constants — must match train_model.py ─────────────────────────────────────
CLASS_TO_IDX  = {'ai': 0, 'real': 1}
IDX_TO_CLASS  = {v: k for k, v in CLASS_TO_IDX.items()}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Confidence below this → prediction is flagged 'Uncertain'
DEFAULT_CONFIDENCE_THRESHOLD = 0.65


# ── Transforms ─────────────────────────────────────────────────────────────────
# Primary inference transform — identical to val_transform in train_model.py
_base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# TTA transforms — multiple views of the same image whose predictions are averaged.
# Each one is a plausible crop/flip the model might encounter in the wild.
_tta_transforms = [
    _base_transform,   # clean centre crop (same as val)
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),   # always flip
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.FiveCrop(224),                 # returns a tuple of 5 crops
        transforms.Lambda(lambda crops: torch.stack([
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(
                transforms.ToTensor()(c)
            ) for c in crops
        ])),
    ]),
]


# ── Detector ───────────────────────────────────────────────────────────────────
class AIImageDetector:
    
    def __init__(
        self,
        model_path: str,
        device: str = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        use_tta: bool = True,
    ):
       
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.confidence_threshold = confidence_threshold
        self.use_tta              = use_tta

        # ── Build model (same architecture as training) ────────────────────────
        self.model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, 2)
        self.model = self.model.to(self.device)

        # ── Load trained weights ───────────────────────────────────────────────
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"Loaded weights: {model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Run train_model.py first to generate it."
            )
        except Exception as e:
            raise RuntimeError(f"Could not load weights from {model_path}: {e}")

        self.model.eval()

    # ── Single image prediction ────────────────────────────────────────────────
    def predict(self, image_path: str) -> dict | None:
    
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Could not open image '{image_path}': {e}")
            return None

        try:
            probs = self._get_probabilities(image)
        except Exception as e:
            print(f"Error during inference on '{image_path}': {e}")
            return None

        return self._build_result(probs)

    # ── Batch prediction ───────────────────────────────────────────────────────
    def predict_batch(self, image_paths: list[str]) -> list[dict | None]:
     
        results = []
        for path in image_paths:
            results.append(self.predict(path))
        return results

    # ── Internal helpers ───────────────────────────────────────────────────────
    def _get_probabilities(self, image: Image.Image) -> torch.Tensor:
       
        if not self.use_tta:
            tensor = _base_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return torch.softmax(self.model(tensor), dim=1)[0]

        all_probs = []

        with torch.no_grad():
            for tfm in _tta_transforms:
                transformed = tfm(image)

                # FiveCrop returns a (5, C, H, W) tensor — handle separately
                if transformed.dim() == 4:
                    batch  = transformed.to(self.device)          # (5, C, H, W)
                    logits = self.model(batch)                    # (5, 2)
                    probs  = torch.softmax(logits, dim=1).mean(0) # (2,)
                else:
                    tensor = transformed.unsqueeze(0).to(self.device)
                    probs  = torch.softmax(self.model(tensor), dim=1)[0]

                all_probs.append(probs)

        return torch.stack(all_probs).mean(0)   # average across all TTA views

    def _build_result(self, probs: torch.Tensor) -> dict:
        """Turn a probability tensor into the result dict."""
        prob_ai   = probs[CLASS_TO_IDX['ai']].item()
        prob_real = probs[CLASS_TO_IDX['real']].item()
        is_ai     = prob_ai > prob_real
        confidence = prob_ai if is_ai else prob_real
        uncertain  = confidence < self.confidence_threshold

        if uncertain:
            label = 'Uncertain'
        else:
            label = 'AI-Generated' if is_ai else 'Real Photo'

        return {
            'prediction':       label,
            'is_ai_generated':  is_ai,
            'confidence':       confidence,
            'probability_ai':   prob_ai,
            'probability_real': prob_real,
            'uncertain':        uncertain,
        }


# ── Convenience wrapper ────────────────────────────────────────────────────────
def get_cnn_results(
    image_path: str,
    model_path: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    use_tta: bool = True,
) -> dict | None:
    
    detector = AIImageDetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        use_tta=use_tta,
    )
    return detector.predict(image_path)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MODEL_PATH = r"C:\Users\Dylan\.vscode\AI Image Detector\layers\cnn_ai_detector.pth"
    IMAGE_PATH = r"C:\Users\Dylan\.vscode\AI Image Detector\data\ai\Cow.png"

    result = get_cnn_results(IMAGE_PATH, model_path=MODEL_PATH)

    if result:
        print("\nCNN Detection Results:")
        print(f"  Prediction:       {result['prediction']}")
        print(f"  Confidence:       {result['confidence']:.2%}")
        print(f"  Probability AI:   {result['probability_ai']:.2%}")
        print(f"  Probability Real: {result['probability_real']:.2%}")
        if result['uncertain']:
            print("Low confidence — treat this result with caution.")