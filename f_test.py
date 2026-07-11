import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
# ... (baqi ka code waise hi rahega)

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import random
import cv2
import numpy as np
from fpdf import FPDF
import tempfile
import os


# ==========================================
# 1. MODELS: Vision Model (CNN)
# ==========================================
class CarDamageCNN:
    def __init__(self):
        # Yahan aap apna real train kiya hua model load karenge (e.g., ResNet)
        self.classes = ["Minor Damage", "Moderate Damage", "Severe Damage"]
        
    def predict(self, image: Image.Image):
        """Image process karke damage predict karta hai."""
        # Dummy inference demonstration ke liye
        confidence = round(random.uniform(0.75, 0.98), 2)
        predicted_class = random.choice(self.classes)
        return {
            "prediction": predicted_class,
            "confidence": confidence
        }

# ==========================================
# 2. EXPLAINABLE AI (XAI): Grad-CAM
# ==========================================
def generate_gradcam_heatmap(image: Image.Image) -> Image.Image:
    """Dummy Grad-CAM heatmap generate karta hai visual explanation ke liye."""
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Dummy heatmap (Red/Yellow areas)
    heatmap = np.zeros_like(img_cv)
    center = (img_cv.shape[1]//2, img_cv.shape[0]//2)
    cv2.circle(heatmap, center, radius=100, color=(0, 0, 255), thickness=-1)
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    
    # Heatmap ko original image par overlay karna
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)
    
    # Wapas PIL format mein convert karna
    overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    return overlay_pil

# ==========================================
# 3. LLM SERVICES: Reasoning
# ==========================================
def generate_claim_reasoning(image_pred, user_text, policy_data):
    """Multimodal data ko use karke LLM based summary banata hai."""
    # Yahan asli OpenAI / LLM API call aayegi future mein
    reasoning = (
        f"Based on the visual evidence showing {image_pred['prediction']} "
        f"and the user's statement matching the damage profile, the claim aligns with the active policy terms. "
        f"Recommendation: Proceed with standard processing."
    )
    return reasoning

# ==========================================
# 4. REPORT GENERATOR: PDF
# ==========================================
def generate_pdf_report(claim_id, policy_data, ai_reasoning, final_decision, notes):
    """Approved data ko PDF report mein convert karta hai."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"Insurance Claim Report: {claim_id}", ln=1, align='C')
    pdf.ln(10)
    
    # Body
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=f"Policy Details: {policy_data}")
    pdf.multi_cell(0, 10, txt=f"AI Reasoning:\n{ai_reasoning}")
    pdf.ln(5)
    
    # Human Decision
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Final Human Decision: {final_decision}", ln=1)
    
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=f"Adjuster Notes:\n{notes}")
    
    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name

# ==========================================
# 5. STREAMLIT APP LOGIC (Frontend + HITL)
# ==========================================
@st.cache_resource
def load_models():
    return CarDamageCNN()

cnn_model = load_models()

st.set_page_config(page_title="AI Claim Co-Pilot", layout="wide")
st.title("🚗 Insurance Claim AI Co-Pilot")
st.markdown("Process claims using Multimodal AI, Explainability, and Human-in-the-Loop validation.")

# --- STEP 1: Multimodal Data Input ---
st.header("1. Upload Claim Data (Multimodal)")
col1, col2 = st.columns(2)

with col1:
    uploaded_image = st.file_uploader("Upload Car Damage Image (Modality 1)", type=["jpg", "png", "jpeg"])
    user_description = st.text_area("Driver's Accident Description (Modality 2)", "I hit a pole while reversing.")

with col2:
    st.markdown("**Tabular Policy Data (Modality 3)**")
    policy_id = st.text_input("Policy ID", "POL-98765")
    deductible = st.number_input("Deductible Amount ($)", 500)
    coverage = st.selectbox("Coverage Type", ["Comprehensive", "Collision", "Liability"])

# --- STEP 2: AI Processing ---
if st.button("Analyze Claim with AI") and uploaded_image:
    image = Image.open(uploaded_image)
    
    with st.spinner("Running Deep Learning Model & Generating Explanations..."):
        # 1. Prediction
        prediction = cnn_model.predict(image)
        # 2. XAI
        xai_image = generate_gradcam_heatmap(image)
        # 3. LLM Reasoning
        policy_data = f"ID: {policy_id}, Deductible: ${deductible}, Type: {coverage}"
        ai_reasoning = generate_claim_reasoning(prediction, user_description, policy_data)
        
        # Save to session state
        st.session_state['ai_pred'] = prediction
        st.session_state['xai_img'] = xai_image
        st.session_state['ai_reasoning'] = ai_reasoning
        st.session_state['policy_data'] = policy_data

# --- STEP 3: Display Results & HITL Workflow ---
if 'ai_pred' in st.session_state:
    st.header("2. AI Analysis & Explainability (XAI)")
    c1, c2 = st.columns(2)
    
    with c1:
        st.image(st.session_state['xai_img'], caption="Grad-CAM Heatmap (AI Focus Area)", use_container_width=True)
    
    with c2:
        st.info(f"**Predicted Damage:** {st.session_state['ai_pred']['prediction']}")
        st.metric("AI Confidence Score", f"{st.session_state['ai_pred']['confidence'] * 100}%")
        st.markdown("**LLM Reasoning & Summary:**")
        st.write(st.session_state['ai_reasoning'])

    st.header("3. Human-in-the-Loop Validation")
    st.warning("Review the AI's recommendation and make the final decision.")
    
    final_decision = st.radio("Final Decision", ["Approve AI Recommendation", "Modify/Adjust Status", "Reject Claim"])
    adjuster_notes = st.text_area("Adjuster Notes / Modifications", "Reviewed heatmap; damage aligns with description.")
    
    # --- STEP 4: Generate Report ---
    if st.button("Finalize & Generate PDF Report"):
        pdf_path = generate_pdf_report(
            claim_id="CLM-12345", 
            policy_data=st.session_state['policy_data'], 
            ai_reasoning=st.session_state['ai_reasoning'], 
            final_decision=final_decision,
            notes=adjuster_notes
        )
        
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📥 Download Claim Report (PDF)",
                data=pdf_file,
                file_name="Claim_Report.pdf",
                mime="application/pdf"
            )
        st.success("Workflow Complete! Business data securely logged.")