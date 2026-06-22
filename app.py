import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import tifffile as tiff
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

st.set_page_config(page_title="Team CodeDevz: Geo-Spatial Dashboard", layout="wide")

# 1. Dashboard Branding
st.title("🌍 Geo Snap: Multi-Spectral Satellite Analysis")
st.write("Created by **Team CodeDevz** | Powered by EfficientNet-B2")
st.markdown("---")

classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
           'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# 2. EfficientNet-B2 Model Loaders
@st.cache_resource
def load_rgb_model():
    model = models.efficientnet_b2(weights=None)
    # EfficientNet classifier is a Sequential block; we modify the Linear layer at index 1
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 10)
    try:
        model.load_state_dict(torch.load('efficientnet_b2_rgb.pth', map_location='cpu'))
    except Exception as e:
        st.sidebar.error("Could not find 'efficientnet_b2_rgb.pth' in the directory.")
    model.eval()
    return model

@st.cache_resource
def load_multispectral_model():
    model_13b = models.efficientnet_b2(weights=None)
    
    # Modify first Conv2d layer for 13 channels (EfficientNet features[0][0] is the conv layer)
    # Default is Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model_13b.features[0][0] = torch.nn.Conv2d(13, 32, kernel_size=3, stride=2, padding=1, bias=False)
    
    # Modify classifier for 10 classes
    model_13b.classifier[1] = torch.nn.Linear(model_13b.classifier[1].in_features, 10)
    
    try:
        model_13b.load_state_dict(torch.load('efficientnet_b2_ms.pth', map_location='cpu'))
    except Exception as e:
        st.sidebar.error("Could not find 'efficientnet_b2_ms.pth' in the directory.")
    model_13b.eval()
    return model_13b

# Initialize Models
rgb_model = load_rgb_model()
multi_model = load_multispectral_model()

# 3. File Upload Interface
uploaded_file = st.file_uploader("Upload a satellite image (.jpg, .jpeg, or .tif)", type=["jpg", "jpeg", "tif"])

if uploaded_file is not None:
    col1, col2, col3 = st.columns(3)
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    input_tensor = None
    display_img = None
    active_model = None
    
    # 4. Routing & Preprocessing
    if file_ext in ['jpg', 'jpeg']:
        active_model = rgb_model
        with col1:
            st.subheader("📷 Input Image (RGB)")
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, use_container_width=True)
            
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(img).unsqueeze(0)
            display_img = np.array(img.resize((64, 64))) / 255.0

    elif file_ext == 'tif':
        active_model = multi_model
        with col1:
            st.subheader("🛰️ Input Image (13-Band)")
            tif_array = tiff.imread(uploaded_file).astype(np.float32)
            
            if tif_array.shape[0] == 13:
                tif_array = np.transpose(tif_array, (1, 2, 0))
            
            # Extract RGB for web preview
            rgb_preview = tif_array[:, :, :3]
            rgb_preview = (rgb_preview - np.min(rgb_preview)) / (np.max(rgb_preview) - np.min(rgb_preview) + 1e-8)
            preview_img = Image.fromarray((rgb_preview * 255).astype(np.uint8)).resize((64, 64))
            
            st.image(preview_img, caption="True-Color UI Preview", use_container_width=True)
            st.info("ℹ️ Analyzing all 13 Multispectral Bands.")
            
            input_tensor = torch.from_numpy(tif_array).permute(2, 0, 1).unsqueeze(0)
            display_img = np.array(preview_img) / 255.0

    # 5. Predictions & Output
    if input_tensor is not None and active_model is not None:
        with col2:
            st.subheader("📈 Class Probabilities")
            with torch.no_grad():
                outputs = active_model(input_tensor)
                probs = F.softmax(outputs, dim=1)[0]
                top_prob, top_idx = torch.max(probs, 0)
                predicted_class = classes[top_idx.item()]
                
            st.success(f"**Prediction:** `{predicted_class}`")
            st.metric(label="Confidence", value=f"{top_prob.item()*100:.2f}%")
            
            prob_dict = {classes[i]: float(probs[i].item()) for i in range(10)}
            st.bar_chart(prob_dict)

        # 6. Grad-CAM for EfficientNet
        with col3:
            st.subheader("🔍 Spatial Focus Map")
            
            try:
                # In EfficientNet, features[-1] is the final Conv block before pooling
                target_layers = [active_model.features[-1]]
                cam = GradCAM(model=active_model, target_layers=target_layers)
                
                grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
                visualization = show_cam_on_image(display_img, grayscale_cam, use_rgb=True)
                
                st.image(visualization, caption=f"Attention Map: {predicted_class}", use_container_width=True)
            except Exception as e:
                st.error(f"Grad-CAM error: {e}")