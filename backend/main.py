from fastapi import FastAPI, File, UploadFile
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
import torchvision.models as models

app = FastAPI()

device = torch.device("cpu")

# -------- LOAD CLASSIFICATION MODEL --------
cls_model = models.efficientnet_b0(pretrained=False)
cls_model.classifier[1] = torch.nn.Linear(1280, 5)
cls_model.load_state_dict(torch.load("models/dr_model.pth", map_location=device))
cls_model.eval()

# -------- LOAD SEGMENTATION MODEL --------
seg_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    classes=1,
    activation=None
)
seg_model.load_state_dict(torch.load("models/dr_segmentation.pth", map_location=device))
seg_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

labels = [
    "Mild",
    "Moderate",
    "No DR",
    "Proliferative DR",
    "Severe"
]

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

        probs = torch.softmax(output, dim=1)
        cls_pred = torch.argmax(probs, dim=1).item()

        print("Pred index:", cls_pred)

    prediction = labels[cls_pred]   

    # ---------- SEGMENTATION ----------
    img_seg = cv2.resize(img_rgb, (224,224))
    img_norm = img_seg / 255.0
    img_norm = np.transpose(img_norm, (2,0,1))

    tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_mask = torch.sigmoid(seg_model(tensor)).squeeze().numpy()

    pred_binary = (pred_mask > 0.6).astype(np.uint8)

    return {
        "prediction": labels[cls_pred],
        "mask": pred_binary.tolist()
    }