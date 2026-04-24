from fastapi import FastAPI, File, UploadFile
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
import torchvision.models as models
from fastapi.middleware.cors import CORSMiddleware
import base64

app = FastAPI()
device = torch.device("cpu")

# -------- CORS --------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- CLASSIFICATION MODEL --------
cls_model = models.efficientnet_b0(pretrained=False)
cls_model.classifier[1] = torch.nn.Linear(1280, 5)
cls_model.load_state_dict(torch.load("models/dr_model.pth", map_location=device))
cls_model.eval()

# -------- SEGMENTATION MODEL --------
seg_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    classes=1,
    activation=None
)
seg_model.load_state_dict(torch.load("models/dr_segmentation.pth", map_location=device))
seg_model.eval()

# -------- HELPERS --------
def apply_fundus_mask(img):
    h, w = img.shape[:2]
    center = (w//2, h//2)
    radius = int(min(center)*0.95)

    mask = np.zeros((h,w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, -1)

    return img * mask[:, :, None]

def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    enhanced = cv2.merge((l,a,b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

labels = ["Mild","Moderate","No DR","Proliferative DR","Severe"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------- CLASSIFICATION ----------
    img_cls = transform(img_rgb).unsqueeze(0)

    with torch.no_grad():
        output = cls_model(img_cls)
        cls_pred = torch.argmax(output, dim=1).item()

    prediction = labels[cls_pred]

    # ---------- SEGMENTATION ----------
    img_seg = cv2.resize(img_rgb, (224,224))
    img_seg = apply_fundus_mask(img_seg)
    img_seg = enhance_contrast(img_seg)

    img_norm = img_seg.astype(np.float32) / 255.0
    img_norm = (img_norm - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
    img_norm = np.transpose(img_norm, (2,0,1))

    tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = seg_model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    # 🔥 adaptive threshold (better than fixed)
    thresh = max(0.15, np.mean(probs) * 1.2)
    pred_binary = (probs > thresh).astype(np.uint8)

    # ---------- MORPHOLOGY ----------
    kernel = np.ones((3,3), np.uint8)
    pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_OPEN, kernel)
    pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_CLOSE, kernel)

    # ---------- CONNECTED COMPONENTS ----------
    num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(pred_binary)

    clean_mask = np.zeros_like(pred_binary)
    lesions = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # 🔥 tighter filtering (fix edge blobs)
        if 30 < area < 2000:
            clean_mask[labels_cc == i] = 1

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            cx = int(x + w / 2)
            cy = int(y + h / 2)
            radius = int(max(w, h) / 2)

            lesions.append({
                "x": cx,
                "y": cy,
                "radius": radius
            })

    # ---------- OVERLAY (FIX: use original image, not masked) ----------
    overlay = img_rgb.copy()

    color_map = {
        "Mild": (255,255,0),
        "Moderate": (255,165,0),
        "Severe": (255,0,0),
        "Proliferative DR": (139,0,0),
        "No DR": (0,255,0)
    }

    color = color_map[prediction]

    for lesion in lesions:
        cv2.circle(
            overlay,
            (lesion["x"], lesion["y"]),
            lesion["radius"],
            color,
            -1
        )

    overlay = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)

    # ---------- ENCODE ----------
    _, overlay_buf = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    overlay_base64 = base64.b64encode(overlay_buf).decode('utf-8')

    return {
        "prediction": prediction,
        "mask": clean_mask.tolist(),
        "overlay": overlay_base64,
        "lesions": lesions
    }