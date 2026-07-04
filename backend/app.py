import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== NLTK SETUP ======================
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

# ====================== LOAD MODEL ======================
MODEL_DIR = Path("../model")
if not MODEL_DIR.exists():
    MODEL_DIR = Path("model")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "best_model.pkl")
    vectorizer = joblib.load(MODEL_DIR / "tfidf.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ====================== PREPROCESSING FUNCTION ======================
le = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def transform(text):
    text = re.sub(r'\(.*?\)\s*-', '', text)   # Remove (text) - patterns
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    lemmatized = [le.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(lemmatized)

# ====================== METRICS DATA ======================
metrics_data = {
    "Model": ["LR", "MNB", "LinearSVC"],
    "Test Accuracy": [0.978, 0.923, 0.985],
    "Test F1-Score": [0.978, 0.920, 0.985],
    "CV Mean Accuracy": [0.979, 0.924, 0.987],
}

metrics_df = pd.DataFrame(metrics_data)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("📊 Model Performance")
    st.markdown("### Best Model: **LinearSVC**")
    
    # Styled DataFrame
    styled_df = metrics_df.style\
        .background_gradient(
            cmap='Blues', 
            subset=['Test Accuracy', 'Test F1-Score', 'CV Mean Accuracy']
        )\
        .format(
            "{:.3f}", 
            subset=metrics_df.select_dtypes(include='number').columns
        )\
        .set_properties(**{'text-align': 'center'})

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.markdown("** Model Info **")

    st.info("""
            Three machine learning classifiers—Logistic Regression, Multinomial Naive Bayes, 
            and LinearSVC—were evaluated using 5-fold stratified cross-validation and an independent test set. 
            Among the evaluated models, LinearSVC achieved the best overall performance, with a mean cross-validation 
            accuracy of 98.76% (±0.05%) and a test accuracy of 98.57%. 
            It also obtained the highest precision (98.46%), recall (98.57%), and F1-score (98.51%), 
            demonstrating strong predictive performance and consistent generalization to unseen data.
            """
            )

# ====================== MAIN UI ======================
st.title("📰 Fake News Detector")
st.subheader("Detect Real vs Fake News using Machine Learning")

st.markdown("""
This tool uses **TF-IDF vectorization** + **Support Vector Classifier (LinearSVC)** 
to determine whether a news article is likely **real** or **fake**.
""")

# Input Area
col1, col2 = st.columns([3, 1])

with col1:
    new_text = st.text_area(
        "Enter the news article text:",
        height=320,
        placeholder="Paste the full news article here (headline + body recommended)...",
        help="Minimum 5 words required"
    )

with col2:
    st.markdown("### How it works")
    st.markdown("""
    1. Text is cleaned and lemmatized
    2. Converted to TF-IDF features
    3. Fed into the trained LinearSVC model
    4. Prediction with confidence score
    """)
    
    st.markdown("---")
    st.caption("Best Model: :green[LinearSVC (98.57% Test Accuracy)]")

# Predict Button
if st.button(" Predict", type="primary", use_container_width=True):
    if len(new_text.strip()) < 5:
        st.error(" Please enter at least 5 words for prediction.")
    else:
        with st.spinner("Analyzing the article..."):
            # Preprocess
            cleaned_text = transform(new_text)
            
            # Vectorize and Predict
            vector = vectorizer.transform([cleaned_text])
            prediction = model.predict(vector)[0]

            # Display Result
            if prediction == 0:
                st.error("**THIS NEWS IS FAKE**")
                st.markdown("### The model predicts this article contains **misinformation**.")
            else:
                st.success("**THIS NEWS IS REAL**")
                st.markdown("### The model predicts this article is **genuine**.")

# Footer
st.markdown("---")
st.markdown(
    "**Disclaimer**: This tool is for educational/demo purposes. Always verify news from multiple trusted sources."
)