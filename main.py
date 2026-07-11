import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Car_Insurance_Claim.csv"
IMAGE_DIR = ROOT / "Car-Damages" / "img"
REVIEW_LOG = ROOT / "review_log.csv"

st.set_page_config(page_title="Insurance Copilot", page_icon="🚗", layout="wide")


@st.cache_data(show_spinner=False)
def load_claim_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data(show_spinner=False)
def load_review_log() -> pd.DataFrame:
    if REVIEW_LOG.exists():
        return pd.read_csv(REVIEW_LOG)
    return pd.DataFrame(columns=["timestamp", "claim_id", "decision", "adjusted_recommendation", "reviewer_notes", "ai_recommendation", "risk_score"])


@st.cache_resource(show_spinner=False)
def train_claim_model(df: pd.DataFrame):
    target = "OUTCOME"
    feature_cols = [col for col in df.columns if col not in {"ID", target}]
    X = df[feature_cols].copy()
    y = df[target].astype(int)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", MLPClassifier(hidden_layer_sizes=(48, 24), max_iter=600, random_state=42)),
        ]
    )

    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    # Permutation importance helps explain which structured features influence the decision.
    perm = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring="accuracy")
    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": perm.importances_mean}
    ).sort_values("importance", ascending=False)

    return model, importance_df, accuracy


def build_claim_record(base_record: dict, overrides: dict) -> pd.DataFrame:
    record = {**base_record, **overrides}
    record_df = pd.DataFrame([record])
    return record_df


def calculate_text_risk(text: str) -> float:
    if not text:
        return 0.5
    lower = text.lower()
    positive_terms = [
        "crash",
        "collision",
        "severe",
        "damage",
        "hail",
        "impact",
        "broken",
        "repair",
        "totaled",
        "front",
        "rear",
        "injury",
        "accident",
    ]
    negative_terms = ["minor", "small", "cosmetic", "wear", "normal"]

    score = 0.5
    for term in positive_terms:
        if term in lower:
            score += 0.06
    for term in negative_terms:
        if term in lower:
            score -= 0.04

    return float(np.clip(score, 0.05, 0.95))


def extract_pdf_text(uploaded_pdf) -> str:
    if uploaded_pdf is None or PdfReader is None:
        return ""
    try:
        reader = PdfReader(io.BytesIO(uploaded_pdf.getvalue()))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(page for page in pages if page).strip()
    except Exception:
        return ""


def calculate_pdf_risk(pdf_text: str) -> float:
    if not pdf_text:
        return 0.5
    lower = pdf_text.lower()
    risk_terms = ["estimate", "repair", "suspicious", "injury", "liability", "photos", "damage", "claim"]
    score = 0.45
    for term in risk_terms:
        if term in lower:
            score += 0.05
    return float(np.clip(score, 0.05, 0.95))


def generate_claim_summary(claim_text: str, pdf_text: str, combined_risk: float) -> str:
    if combined_risk >= 0.7:
        action = "high priority escalation"
    elif combined_risk >= 0.45:
        action = "follow-up and document review"
    else:
        action = "standard review"

    evidence = []
    if claim_text:
        evidence.append("claim narrative provided")
    if pdf_text:
        evidence.append("supporting PDF evidence uploaded")
    evidence_text = ", ".join(evidence) if evidence else "limited evidence available"
    return (
        f"This claim should move into {action}. The AI copilot combined structured risk, narrative context, and image evidence "
        f"to estimate a risk level of {combined_risk:.1%}. Evidence status: {evidence_text}."
    )


def calculate_image_risk(image_path: Path) -> float:
    if not image_path.exists():
        return 0.5

    image = Image.open(image_path).convert("RGB")
    arr = np.array(image)
    gray = np.array(image.convert("L"))

    edge_density = np.mean(np.abs(np.diff(gray, axis=0)) > 20)
    contrast = np.std(gray) / 255.0
    saturation = np.std(arr[:, :, 1]) / 255.0
    risk_score = 0.4 * edge_density + 0.35 * contrast + 0.25 * saturation

    return float(np.clip(risk_score, 0.05, 0.95))


def build_pdf_report(result: dict, top_features: pd.DataFrame, review_decision: str, reviewer_notes: str) -> bytes:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
    except Exception:
        return b"%PDF-1.4\n%PDF generation dependency is unavailable.\n"

    lines = [
        "Insurance Claim Copilot Report",
        "==============================",
        f"Claim ID: {result['claim_id']}",
        f"Overall Review Score: {result['combined_risk']:.1%}",
        f"Recommendation: {result['recommendation']}",
        f"Model Confidence: {result['confidence']:.1%}",
        "",
        "Key explainability drivers:",
    ]
    for _, row in top_features.head(6).iterrows():
        lines.append(f"- {row['feature']}: {row['importance']:.3f}")

    lines.extend(["", f"Reviewer decision: {review_decision}", f"Reviewer notes: {reviewer_notes or 'No notes provided'}"])

    buffer = io.BytesIO()
    pdf_canvas = canvas.Canvas(buffer, pagesize=letter)
    pdf_canvas.setTitle("Insurance Claim Copilot Report")
    pdf_canvas.setFont("Helvetica", 12)
    pdf_canvas.setFillColor(colors.black)
    y = 750
    for line in lines:
        pdf_canvas.drawString(50, y, line)
        y -= 14
    pdf_canvas.showPage()
    pdf_canvas.save()
    return buffer.getvalue()


def main() -> None:
    df = load_claim_data()
    model, importance_df, accuracy = train_claim_model(df)

    st.title("🚗 Insurance Claim Copilot")
    st.caption("A multimodal AI assistant for claim triage, damage inspection, human review, and explainable reporting.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training samples", f"{len(df):,}")
    with col2:
        st.metric("Model accuracy", f"{accuracy:.1%}")
    with col3:
        st.metric("Review workflow", "Approve / Reject / Modify")

    st.subheader("1. Structured claim profile")
    with st.sidebar:
        st.header("Claim inputs")
        sample_id = st.selectbox("Select sample profile", df["ID"].tolist(), format_func=lambda x: f"Claim {x}")
        base_record = df.loc[df["ID"] == sample_id].iloc[0].to_dict()

        overrides = {}
        overrides["CREDIT_SCORE"] = st.slider("Credit score", 0.0, 1.0, float(base_record["CREDIT_SCORE"]), 0.01)
        overrides["ANNUAL_MILEAGE"] = st.number_input("Annual mileage", min_value=0, value=int(base_record["ANNUAL_MILEAGE"]), step=1000)
        overrides["SPEEDING_VIOLATIONS"] = st.slider("Speeding violations", 0, 10, int(base_record["SPEEDING_VIOLATIONS"]))
        overrides["DUIS"] = st.slider("DUIs", 0, 5, int(base_record["DUIS"]))
        overrides["PAST_ACCIDENTS"] = st.slider("Past accidents", 0, 10, int(base_record["PAST_ACCIDENTS"]))
        overrides["VEHICLE_TYPE"] = st.selectbox("Vehicle type", options=sorted(df["VEHICLE_TYPE"].dropna().unique()), index=list(sorted(df["VEHICLE_TYPE"].dropna().unique())).index(base_record["VEHICLE_TYPE"]))
        overrides["INCOME"] = st.selectbox("Income tier", options=sorted(df["INCOME"].dropna().unique()), index=list(sorted(df["INCOME"].dropna().unique())).index(base_record["INCOME"]))
        overrides["VEHICLE_OWNERSHIP"] = st.selectbox("Vehicle owned", options=[0.0, 1.0], index=0 if float(base_record["VEHICLE_OWNERSHIP"]) == 0.0 else 1)
        overrides["MARRIED"] = st.selectbox("Married", options=[0.0, 1.0], index=0 if float(base_record["MARRIED"]) == 0.0 else 1)

    feature_cols = [col for col in df.columns if col not in {"ID", "OUTCOME"}]
    record_df = build_claim_record(base_record, overrides)
    record_df = record_df[feature_cols]
    record_df = record_df.astype({col: "float64" for col in record_df.select_dtypes(include=["number"]).columns})

    st.subheader("2. Damage image, PDF evidence, and claim notes")
    left_col, right_col = st.columns(2)
    with left_col:
        sample_image = next(iter(IMAGE_DIR.glob("*.jpg")), None)
        uploaded_image = st.file_uploader("Upload damage image", type=["jpg", "jpeg", "png"])
        uploaded_pdf = st.file_uploader("Upload supporting PDF", type=["pdf"])
        if uploaded_image is not None:
            image_path = ROOT / uploaded_image.name
            image_bytes = uploaded_image.read()
            with open(image_path, "wb") as handle:
                handle.write(image_bytes)
            selected_image = image_path
        else:
            selected_image = sample_image

        if selected_image is not None:
            image = Image.open(selected_image)
            st.image(image, caption="Damage image", use_container_width=True)

    with right_col:
        claim_text = st.text_area(
            "Claim narrative",
            value="A rear bumper collision caused significant damage during a rainy evening. The customer reported a loud impact and pending repair estimate.",
            height=180,
        )
        pdf_text = extract_pdf_text(uploaded_pdf)
        if pdf_text:
            st.text_area("Extracted PDF evidence preview", value=pdf_text[:2500], height=180)
        else:
            st.info("Upload a PDF to add a fourth evidence modality for inspection.")

    st.subheader("3. AI recommendation")
    text_risk = calculate_text_risk(claim_text)
    image_risk = calculate_image_risk(selected_image) if selected_image is not None else 0.5
    pdf_risk = calculate_pdf_risk(pdf_text)
    prediction_prob = model.predict_proba(record_df)[0, 1]
    combined_risk = float(np.clip(0.6 * prediction_prob + 0.15 * text_risk + 0.15 * image_risk + 0.1 * pdf_risk, 0.02, 0.98))
    confidence = max(prediction_prob, 1 - prediction_prob)

    if combined_risk >= 0.7:
        recommendation = "Escalate for human review"
        color = "🔴"
    elif combined_risk >= 0.45:
        recommendation = "Monitor and request additional documents"
        color = "🟠"
    else:
        recommendation = "Approve with standard workflow"
        color = "🟢"

    st.markdown(f"### {color} Recommended action: {recommendation}")
    st.progress(combined_risk)
    st.write(f"Model confidence: {confidence:.1%}")
    st.write(f"Structured risk score: {prediction_prob:.1%}")
    st.write(f"Text narrative risk: {text_risk:.1%}")
    st.write(f"Image damage risk: {image_risk:.1%}")
    st.write(f"PDF evidence risk: {pdf_risk:.1%}")
    st.info(generate_claim_summary(claim_text, pdf_text, combined_risk))

    st.subheader("4. Explainable AI")
    top_features = importance_df.head(8).copy()
    top_features["importance"] = top_features["importance"].round(3)
    st.dataframe(top_features, use_container_width=True)
    st.bar_chart(top_features.set_index("feature")["importance"])

    st.subheader("5. Human-in-the-loop review")
    with st.form("review_form"):
        decision = st.radio("Reviewer decision", ["Approve", "Reject", "Modify"], horizontal=True)
        adjusted_recommendation = st.text_area("Adjust the AI recommendation", value=recommendation)
        reviewer_notes = st.text_area("Reviewer notes")
        submitted = st.form_submit_button("Save review")

    if submitted:
        review_row = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "claim_id": int(sample_id),
            "decision": decision,
            "adjusted_recommendation": adjusted_recommendation,
            "reviewer_notes": reviewer_notes,
            "ai_recommendation": recommendation,
            "risk_score": round(combined_risk, 3),
        }
        if REVIEW_LOG.exists():
            log_df = pd.read_csv(REVIEW_LOG)
        else:
            log_df = pd.DataFrame(columns=review_row.keys())
        log_df = pd.concat([log_df, pd.DataFrame([review_row])], ignore_index=True)
        log_df.to_csv(REVIEW_LOG, index=False)
        st.success("Review saved to local log for auditability.")

    st.subheader("6. Downloadable report")
    report_bytes = build_pdf_report(
        {
            "claim_id": int(sample_id),
            "combined_risk": combined_risk,
            "recommendation": recommendation,
            "confidence": confidence,
        },
        top_features,
        decision if submitted else "Pending",
        reviewer_notes if submitted else "",
    )
    st.download_button(
        label="Download PDF report",
        data=report_bytes,
        file_name="insurance_claim_report.pdf",
        mime="application/pdf",
        key="download_pdf_report",
    )

    st.subheader("7. Business value and commercialization")
    st.markdown(
        """
        - Product: AI copilot for insurers, brokers, and claims adjusters.
        - Revenue model: per-claim SaaS pricing, premium analytics seats, and API integrations.
        - Commercialization path: pilot with a regional insurer, prove faster triage and lower fraud leakage, then expand into auto, health, and property claims.
        - Differentiator: explainable recommendations, human approval, multimodal evidence review, and downloadable audit reports.
        """
    )

    st.subheader("8. Operational workflow")
    st.markdown(
        """
        1. Intake claim details, image evidence, and PDF documents.
        2. Score the claim using structured modeling, text analysis, and image heuristics.
        3. Present the AI recommendation along with explainable drivers for reviewer approval.
        4. Save the final decision and export a PDF audit trail for compliance.
        """
    )

    st.subheader("9. Review history")
    review_log = load_review_log()
    if not review_log.empty:
        st.dataframe(review_log.tail(10), use_container_width=True)
    else:
        st.info("No review decisions have been saved yet. Submit one from the review form to build an audit trail.")

    st.subheader("10. Model and explainability notes")
    st.markdown(
        """
        - Model: a multi-layer perceptron trained on the insurance tabular dataset.
        - Explainability: permutation importance ranks the structured features that most affect the decision.
        - Evidence blending: text and image features are combined to create a richer claim-risk score.
        - Human-in-the-loop: reviewer edits remain visible and auditable for downstream operations.
        """
    )


if __name__ == "__main__":
    main()
