import os
import gdown
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

LABELS_AR = ["طبيعي", "التهاب رئوي", "COVID-19"]

class MedicalClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b4(weights=None)
        in_f = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_f, 512),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 3)
        )
    def forward(self, x): return self.backbone(x)

MODEL_PATH = "best_model.pth"
if not os.path.exists(MODEL_PATH):
    print("جاري تحميل النموذج...")
    gdown.download(
        "https://drive.google.com/uc?id=10knULnWuir91hD4Ai-o5aew91JjhOXWn",
        MODEL_PATH,
        quiet=False
    )

device = torch.device("cpu")
model = MedicalClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    img = Image.open(io.BytesIO(request.files["file"].read())).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].tolist()
    return jsonify({
        "diagnosis": LABELS_AR[probs.index(max(probs))],
        "confidence": round(max(probs) * 100, 1),
        "probabilities": {LABELS_AR[i]: round(p*100, 1) for i, p in enumerate(probs)}
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)