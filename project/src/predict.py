from pathlib import Path

import torch
from PIL import Image

from src.model import build_resnet18
from src.data import get_transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "artifacts" / "model.pth"

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

def load_model():
    model = build_resnet18(num_classes=2)
    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()

_, eval_tf = get_transforms()

@torch.no_grad()
def predict_image(image_path: str):

    image = Image.open(image_path).convert("RGB")

    x = eval_tf(image)
    x = x.unsqueeze(0).to(DEVICE)

    logits = model(x)

    probs = torch.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[0, pred_idx].item()

    return {
        "class": pred_class,
        "confidence": round(confidence, 4)
    }