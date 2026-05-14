import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Skin Cancer AI Diagnostic", layout="wide")

st.title("🩺 Clinical AI Diagnostic Assistant")
st.write("Upload a dermoscopic image to get an AI-powered diagnosis and localization heatmap.")

# 1. Sidebar for Server Status
st.sidebar.header("System Status")
try:
    # Check if FastAPI is running
    requests.get("http://127.0.0.1:8000/docs")
    st.sidebar.success("API Server: Connected")
except:
    st.sidebar.error("API Server: Disconnected (Please run main.py)")

# 2. File Uploader
uploaded_file = st.file_uploader("Choose a lesion image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Run Diagnostic"):
        with st.spinner('Analyzing...'):
            # Convert uploaded file to format FastAPI expects
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            # Send to your FastAPI server
            response = requests.post("http://127.0.0.1:8000/diagnose", files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                with col2:
                    st.subheader(f"Diagnosis: {result['diagnosis']}")
                    st.write(f"**Confidence:** {result['confidence']}")
                    
                    # Fetch the XAI Map from the static URL provided by the API
                    full_url = f"http://127.0.0.1:8000{result['report_url']}"
                    st.image(full_url, caption="AI Interpretability Report", use_column_width=True)
            else:
                st.error("Diagnostic Failed. Please check the API server logs.")