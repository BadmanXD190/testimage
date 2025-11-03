import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # helps avoid rare CPU segfaults

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

st.set_page_config(page_title="Realtime Webcam + Keras", layout="centered")
st.title("Realtime Webcam Classification with Keras")

MODEL_PATH = os.getenv("MODEL_PATH", "keras_model.h5")
LABELS_PATH = os.getenv("LABELS_PATH", "labels.txt")
TARGET_SIZE = (224, 224)

@st.cache_resource
@st.cache_resource
def load_keras_model():
    import tensorflow as tf
    from tensorflow import keras
    from keras.layers import DepthwiseConv2D

    # Shim: drop unknown "groups" argument from older saved configs
    class DepthwiseConv2DCompat(DepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop("groups", None)
            super().__init__(*args, **kwargs)

    # Also handle legacy ReLU6 if your model used MobileNet-style ops
    def relu6(x):
        return keras.activations.relu(x, max_value=6)

    custom_objects = {
        "DepthwiseConv2D": DepthwiseConv2DCompat,
        "relu6": relu6,                # harmless if unused
        "tf": tf,                      # sometimes referenced in Lambda layers
    }

    return keras.models.load_model("keras_model.h5", compile=False, custom_objects=custom_objects)


@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        out = []
        for ln in f.readlines():
            ln = ln.strip()
            parts = ln.split(" ", 1)
            out.append(parts[1] if len(parts) == 2 and parts[0].isdigit() else ln)
        return out


# Try to load model/labels and show helpful hints if missing
model, labels = None, []
model_error = None
try:
    model = load_keras_model()
    labels = load_labels()
except Exception as e:
    model_error = str(e)

def preprocess_frame(frame_rgb: np.ndarray) -> np.ndarray:
    img = Image.fromarray(frame_rgb).convert("RGB")
    img = ImageOps.fit(img, TARGET_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(img).astype(np.float32)
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)

def predict(frame_rgb: np.ndarray):
    data = preprocess_frame(frame_rgb)
    preds = model.predict(data, verbose=0)[0]
    idx = int(np.argmax(preds))
    return (labels[idx] if idx < len(labels) else f"class_{idx}"), float(preds[idx])

def draw_overlay(bgr_frame: np.ndarray, text: str):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick, margin = 0.8, 2, 12
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x1, y1 = margin, margin
    x2, y2 = x1 + tw + 20, y1 + th + 20
    cv2.rectangle(bgr_frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(bgr_frame, text, (x1 + 10, y1 + th + 4), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return bgr_frame

if model_error:
    st.error("Model failed to load: " + model_error)
    st.stop()

st.success("Model and labels loaded.")

# WebRTC config (public STUN)
rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.write("Allow camera access to start realtime inference.")
pred_placeholder = st.empty()

def video_frame_callback(frame: av.VideoFrame):
    img_bgr = frame.to_ndarray(format="bgr24")
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        label, conf = predict(rgb)
        text = f"{label}  {conf:.2f}"
        pred_placeholder.write({"prediction": label, "confidence": round(conf, 4)})
    except Exception as e:
        text = f"Error: {str(e)[:40]}"
    out = draw_overlay(img_bgr, text)
    return av.VideoFrame.from_ndarray(out, format="bgr24")

webrtc_streamer(
    key="keras-realtime",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_config,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)


