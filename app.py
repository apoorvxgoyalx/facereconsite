import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load pre-trained models
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Face detection model
    mtcnn = MTCNN(
        image_size=160, 
        margin=0, 
        min_face_size=20,
        device=device
    )
    
    # Face recognition model
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    return mtcnn, resnet, device

# Super-resolution enhancement using PyTorch
def enhance_image_sr(image):
    """
    Simulate super-resolution enhancement
    Note: Replace with actual super-resolution model in production
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to torch tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Simulate enhancement (basic contrast and sharpness)
    img_enhanced = cv2.detailEnhance(img_array, sigma_s=10, sigma_r=0.15)
    img_enhanced = cv2.edgePreservingFilter(img_enhanced, flags=1, sigma_s=60, sigma_r=0.4)
    
    return Image.fromarray(img_enhanced)

# Face detection and recognition function
def detect_and_recognize_faces(image, mtcnn, resnet, device):
    """
    Detect and recognize faces in the image
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Detect faces
    try:
        faces, probs = mtcnn.detect(img_array, landmarks=False)
        
        # If faces are detected
        if faces is not None:
            # Draw rectangles around faces
            for (x, y, x2, y2), prob in zip(faces, probs):
                cv2.rectangle(
                    img_array, 
                    (int(x), int(y)), 
                    (int(x2), int(y2)), 
                    (0, 255, 0), 
                    2
                )
            
            return Image.fromarray(img_array), len(faces), probs
        else:
            return Image.fromarray(img_array), 0, None
    except Exception as e:
        st.error(f"Error in face detection: {e}")
        return Image.fromarray(img_array), 0, None

# Streamlit App
def main():
    # Set page configuration
    st.set_page_config(
        page_title="FacRecon - Image Enhancement",
        page_icon=":detective:",
        layout="wide"
    )
    
    # Custom CSS for modern, sleek design
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #3494E6, #2980B9);
        color: white;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5em;
    }
    .stButton>button {
        background-color: #3498db !important;
        color: white !important;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9 !important;
        transform: scale(1.05);
    }
    .metric-container {
        background-color: #f7f9fc;
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('''
    <div class="main-header">
        <h1>FacRecon: AI Image Enhancement</h1>
        <p>Convert Low-Quality Evidence to Clear Identification</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Load models
    try:
        mtcnn, resnet, device = load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Low-Quality Image", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image for enhancement and face detection"
    )
    
    # Processing if file is uploaded
    if uploaded_file is not None:
        # Open the image
        original_image = Image.open(uploaded_file)
        
        # Create columns for display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(original_image, use_column_width=True)
        
        with col2:
            st.subheader("Enhanced Image")
            enhanced_image = enhance_image_sr(original_image)
            st.image(enhanced_image, use_column_width=True)
        
        # Face Detection
        st.subheader("Face Detection Results")
        detected_image, face_count, face_probs = detect_and_recognize_faces(
            enhanced_image, mtcnn, resnet, device
        )
        
        # Display detected faces
        st.image(detected_image, use_column_width=True)
        
        # Metrics container
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        
        # Metrics
        col_metrics1, col_metrics2 = st.columns(2)
        
        with col_metrics1:
            st.metric("Faces Detected", face_count)
        
        with col_metrics2:
            if face_probs is not None:
                avg_confidence = np.mean(face_probs) * 100
                st.metric("Detection Confidence", f"{avg_confidence:.2f}%")
            else:
                st.metric("Detection Confidence", "N/A")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Placeholder content
        st.info("""
        ### Welcome to FacRecon AI Enhancement
        
        🔍 Upload an image to:
        - Enhance Image Quality
        - Detect Faces
        - Improve Evidence Clarity
        
        Our AI transforms unclear images into actionable intelligence.
        """)

# Run the app
if __name__ == "__main__":
    main()
