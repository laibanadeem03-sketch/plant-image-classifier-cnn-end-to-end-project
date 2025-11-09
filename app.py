import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="LeafGuard: Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Load External CSS ---
def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:   # 👈 Added encoding
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css('style.css')

# --- Background Image Setup ---
# 🔹 CHANGE THIS PATH according to your system
background_path = r"D:\Documents\7th Semester\DL\Project\LeafGuard_App\assets\background.png"

if os.path.exists(background_path):
    bg_image_url = f"file://{background_path.replace(os.sep, '/')}"
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("{bg_image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Model and Class Names ---
MODEL_PATH = 'models/best_custom_cnn_model.keras'

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# --- Disease Descriptions + Cures ---
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "description": "Apple scab causes dark, scabby lesions on leaves and fruits, reducing quality.",
        "cure": "Remove fallen leaves, prune trees for airflow, and spray fungicides like mancozeb."
    },
    "Apple___Black_rot": {
        "description": "Black rot leads to leaf spots and fruit decay caused by Botryosphaeria obtusa fungus.",
        "cure": "Remove infected branches, disinfect pruning tools, and apply thiophanate-methyl fungicide."
    },
    "Tomato___Late_blight": {
        "description": "Late blight spreads fast in humid weather, causing dark patches and fruit rot.",
        "cure": "Destroy infected plants, avoid overhead watering, and spray with copper-based fungicides."
    },
    "Strawberry___Leaf_scorch": {
        "description": "Leaf scorch causes browning and drying of leaf edges, reducing plant vigor.",
        "cure": "Avoid overhead watering, remove infected leaves, and improve air circulation."
    },
    "Default": {
        "description": "No detailed data found for this disease.",
        "cure": "Consult a local agricultural expert for a suitable fungicide or remedy."
    }
}

# --- Load Model ---
@st.cache_resource
def load_model(path):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(MODEL_PATH)

# --- Image Preprocessing ---
def preprocess_image(image):
    img = image.resize((64, 64))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# --- Prediction Parser ---
def parse_prediction(pred_class):
    parts = pred_class.split('___')
    plant = parts[0].replace('_', ' ')
    disease = parts[1].replace('_', ' ')
    status = "Healthy" if "healthy" in disease.lower() else "Diseased"
    return plant, disease, status

# --- Main Title ---
st.title("🌿 LeafGuard: Plant Disease Detector")
st.markdown("Upload an image of a plant leaf, and the AI will identify the plant and detect potential diseases.")

# --- Sidebar ---
# 🔹 CHANGE PATH BELOW for your logo
st.sidebar.image(r"D:\Documents\7th Semester\DL\Project\LeafGuard_App\assets\logo.png", use_container_width=True)

st.sidebar.header("About LeafGuard")
st.sidebar.info(
    "This application uses a Deep Learning model (CNN) "
    "to detect diseases in plant leaves. Trained on the PlantVillage dataset, "
    "it can identify 38 different classes of plant health."
)
st.sidebar.header("How to Use")
st.sidebar.markdown("""
1. **Upload an Image:** Click the 'Browse files' button.  
2. **Select a Leaf Image:** Use a clear photo of a single leaf.  
3. **View Results:** The model predicts the plant and disease condition.
""")

# --- File Upload ---
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption='Uploaded Leaf Image', use_container_width=True)

    with col2:
        with st.spinner('AI is analyzing...'):
            if model:
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                predicted_class_index = np.argmax(prediction)
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                confidence = np.max(prediction) * 100

                plant, disease, status = parse_prediction(predicted_class_name)

                st.subheader("🔎 Analysis Results")
                st.metric(label="Predicted Plant", value=plant)

                if status == "Healthy":
                    st.success(f"Status: {status}")
                else:
                    st.error(f"Status: {status}")

                st.metric(label="Detected Issue", value=disease)
                st.write("Confidence Level:")
                st.progress(int(confidence))

                with st.expander("🌱 Learn more about this condition..."):
                    info = DISEASE_INFO.get(predicted_class_name, DISEASE_INFO["Default"])
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**Cure/Treatment:** {info['cure']}")
            else:
                st.error("Model not loaded. Please check the model path.")

st.markdown("---")
st.markdown("Project by **Iqra, Sameen & Laiba** | Powered by **Streamlit & TensorFlow**")
