import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import tifffile as tiff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from datetime import datetime

st.set_page_config(
    page_title="GeoSnap",
    layout="wide"
)

st.markdown("""<style>.block-container{    padding-top:2rem;    padding-bottom:2rem;    max-width:1200px;}[data-testid="stSidebar"]{    min-width:280px;    max-width:280px;}@media (max-width:768px){    [data-testid="stSidebar"]{        min-width:100%;        max-width:100%;    }    .block-container{        padding-left:1rem;        padding-right:1rem;    }}</style>""", unsafe_allow_html=True)

# ---------------------------------------------------
# Header
# ---------------------------------------------------

st.title("GeoSnap")
st.subheader(
    "Land Use Classification and Environmental Insights from Satellite Imagery"
)

st.markdown("""
Developed by **PixelPioneers**

Supported inputs:

• RGB Images (.jpg, .jpeg, .png, .bmp, .webp)

• Sentinel-2 Multispectral Images (.tif, .tiff)

The application automatically detects the uploaded file type and loads the appropriate model.
""")

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:

    st.header("About")

    st.markdown("""
### Team
PixelPioneers

### Models
**EfficientNet-B2 RGB**
• Accuracy: 97.33%

**EfficientNet-B2 Multispectral**
• Accuracy: 98.79%

[📓 View Google Colab Notebook](https://colab.research.google.com/)

### Supported Formats

RGB Images
• JPG
• JPEG
• PNG
• BMP
• WEBP

Multispectral Images
• TIF
• TIFF (13 Bands)

### Features
• Land Use Classification
• Environmental Analysis
• Probability Distribution
• Downloadable Reports
""")

# ---------------------------------------------------
# RGB Transform
# ---------------------------------------------------

rgb_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------------------------------
# Model Loaders
# ---------------------------------------------------

@st.cache_resource
def load_rgb_model():

    ckpt = torch.load(
        "efficientnet_b2_rgb_final.pth",
        map_location="cpu"
    )

    classes = ckpt["classes"]

    model = models.efficientnet_b2(
        weights=None
    )

    n_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(
            n_features,
            len(classes)
        )
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.eval()

    return model, classes


@st.cache_resource
def load_ms_model():

    ckpt = torch.load(
        "efficientnet_b2_ms_final.pth",
        map_location="cpu"
    )

    classes = ckpt["classes"]

    model = models.efficientnet_b2(
        weights=None
    )

    old = model.features[0][0]

    model.features[0][0] = nn.Conv2d(
        13,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False
    )

    n_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(
            n_features,
            len(classes)
        )
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.eval()

    return (
        model,
        classes,
        ckpt["band_mean"],
        ckpt["band_std"]
    )

# ---------------------------------------------------
# File Upload
# ---------------------------------------------------

uploaded = st.file_uploader(
    "Upload Satellite Image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp",
        "tif",
        "tiff"
    ]
)

if uploaded:

    name = uploaded.name.lower()

    rgb_ext = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    tif_ext = (
        ".tif",
        ".tiff"
    )

    # ===============================================
    # RGB MODEL
    # ===============================================

    if name.endswith(rgb_ext):

        st.info(
            "RGB image detected. Using RGB classification model."
        )

        model, classes = load_rgb_model()

        image = Image.open(
            uploaded
        ).convert("RGB")

        x = rgb_transform(
            image
        ).unsqueeze(0)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(
                logits,
                dim=1
            )

    # ===============================================
    # TIFF MODEL
    # ===============================================

    else:

        st.info(
            "Multispectral image detected. Using 13-band model."
        )

        (
            model,
            classes,
            band_mean,
            band_std
        ) = load_ms_model()

        img = tiff.imread(uploaded).astype(np.float32)
        if img.ndim != 3:
            st.error(f"Expected a 3D TIFF, got shape {img.shape}")
            st.stop()
        if img.shape[0] == 13:
            pass
        elif img.shape[-1] == 13:
            img = np.transpose(img, (2, 0, 1))
        else:
            st.error(f"Expected 13 bands, got shape {img.shape}")
            st.stop()
        band_mean = np.array(band_mean, dtype=np.float32)
        band_std = np.array(band_std, dtype=np.float32)
        img = (
            img - band_mean[:, None, None]) / (
            band_std[:, None, None] + 1e-8)

        x = torch.tensor(
            img,
            dtype=torch.float32
        ).unsqueeze(0)

        col1, col2 = st.columns(2)

        with col1:

            rgb_preview = np.stack([
                img[3],
                img[2],
                img[1]
            ], axis=-1)

            rgb_preview = (
                rgb_preview
                - rgb_preview.min()
            ) / (
                rgb_preview.max()
                - rgb_preview.min()
                + 1e-8
            )

            st.image(
                rgb_preview,
                caption="Preview"
            )

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(
                logits,
                dim=1
            )

    # ===============================================
    # Prediction
    # ===============================================

    pred = torch.argmax(
        probs,
        dim=1
    ).item()

    predicted = classes[pred]

    confidence = probs[0][pred].item()

    with col2:

        st.subheader("Prediction")

        st.success(
            predicted
        )

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        st.subheader(
            "Top Predictions"
        )

        top_probs, top_idx = torch.topk(
            probs,
            min(
                3,
                len(classes)
            )
        )

        for p, i in zip(
            top_probs[0],
            top_idx[0]
        ):
            st.write(
                f"{classes[i]} : {p.item()*100:.2f}%"
            )

    # ===============================================
    # Probability Chart
    # ===============================================

    st.subheader(
        "Probability Distribution"
    )

    chart = pd.DataFrame({
        "Class": classes,
        "Probability":
            probs[0].numpy()
    })

    st.bar_chart(
        chart.set_index(
            "Class"
        )
    )

    # ===============================================
    # TASK 3
    # ===============================================

    if name.endswith(tif_ext):

        st.subheader(
            "Environmental Analysis"
        )

        nir = img[7]
        red = img[3]
        green = img[2]
        swir = img[10]

        ndvi = (
            nir - red
        ) / (
            nir + red + 1e-8
        )

        ndwi = (
            green - nir
        ) / (
            green + nir + 1e-8
        )

        moisture = (
            nir - swir
        ) / (
            nir + swir + 1e-8
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Average NDVI",
            f"{ndvi.mean():.3f}"
        )

        c2.metric(
            "Average NDWI",
            f"{ndwi.mean():.3f}"
        )

        c3.metric(
            "Moisture Index",
            f"{moisture.mean():.3f}"
        )

        fig, ax = plt.subplots(
            1,
            3,
            figsize=(15, 4)
        )

        ax[0].imshow(
            ndvi,
            cmap="RdYlGn"
        )
        ax[0].set_title(
            "NDVI"
        )

        ax[1].imshow(
            ndwi,
            cmap="Blues"
        )
        ax[1].set_title(
            "NDWI"
        )

        ax[2].imshow(
            moisture,
            cmap="viridis"
        )
        ax[2].set_title(
            "Moisture"
        )

        for a in ax:
            a.axis("off")

        st.pyplot(fig)

    # ===============================================
    # Report
    # ===============================================

    report = f"""
GeoSnap Classification Report

Prediction:
{predicted}

Confidence:
{confidence*100:.2f} %

Timestamp:
{datetime.now()}

Generated by:
PixelPioneers
"""

    st.download_button(
        "Download Report",
        report,
        file_name="geosnap_report.txt"
    )