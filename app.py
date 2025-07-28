import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

class_names = ['Normal', 'Pneumonia']

@st.cache_resource
def load_model(path=r"D:/Projects/Healthcare/DL/model/modelchest_xray_model.pth"):
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, len(class_names))
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def predict(image, model):
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485],[0.229])
    ])
    x = tf(image).unsqueeze(0)
    out = model(x)
    return class_names[out.argmax(1).item()]

st.title("🩺 Chest X‑ray Classifier")
uploaded = st.file_uploader("Upload a chest X‑ray…", type=["jpg","png","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)
    if st.button("Predict"):
        model = load_model()
        label = predict(img, model)
        st.success(f"**{label}**")
