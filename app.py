import streamlit as st
import joblib
import numpy as np

# Load trained model, TF-IDF vectorizer, and scaler
model = joblib.load("slogan_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("scaler.pkl")

# Function to predict slogan quality
def predict_slogan(slogan, raw_revenue, identical_trademarks, similar_trademarks, social_media_taken):
    # Transform the slogan using the loaded TF-IDF vectorizer
    slogan_tfidf = vectorizer.transform([slogan]).toarray()
    
    # Scale numeric inputs
    numeric_features = np.array([[raw_revenue, identical_trademarks, similar_trademarks, social_media_taken]])
    numeric_scaled = scaler.transform(numeric_features)
    
    # Combine text and numeric features
    X_input = np.hstack((slogan_tfidf, numeric_scaled))
    
    # Make prediction
    prediction = model.predict(X_input)[0]
    
    # Label based on threshold
    label = "🌟 Strong & Memorable Slogan!" if prediction > 0.5 else "⚠️ Consider Refining Your Slogan"
    
    return prediction, label

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
        }
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #2C3E50;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #7F8C8D;
        }
        .stTextInput, .stSlider, .stRadio {
            font-size: 16px !important;
        }
        .result-box {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: #2C3E50;
            margin-top: 20px;
        }
        .center-button {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown("<h1 class='title'>Slogan Quality Predictor 🚀</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter your slogan and business details to assess its quality.</p>", unsafe_allow_html=True)

# User input fields
slogan = st.text_input("📢 Enter Your Slogan:", placeholder="E.g., Just Do It")

# Numeric feature inputs
raw_revenue = st.slider(
    "💰 Estimated Business Revenue (0-100)",
    min_value=0, max_value=100, value=50,
    help="How much revenue does the business generate? (0 = No revenue, 100 = Very high revenue)"
)

col1, col2 = st.columns(2)
with col1:
    identical_trademarks = st.radio("⚖️ Identical trademarks?", ["No", "Yes"], help="Are there identical trademarks?")
    similar_trademarks = st.radio("🔍 Similar trademarks?", ["No", "Yes"], help="Are there similar trademarks?")
with col2:
    social_media_taken = st.radio("📱 Social media handle taken?", ["No", "Yes"], help="Is the username taken?")

identical_trademarks = 0 if identical_trademarks == "Yes" else 1
similar_trademarks = 0 if similar_trademarks == "Yes" else 1
social_media_taken = 0 if social_media_taken == "Yes" else 1

# Centered Predict Button
predict_clicked = st.button("🔍 Check Slogan Quality")

if predict_clicked:
    if slogan.strip():
        score, label = predict_slogan(slogan, raw_revenue, identical_trademarks, similar_trademarks, social_media_taken)

        # Display results with styling
        st.markdown(f"""
            <div class="result-box">
                <p>{label}</p>
                <p>🎯 Prediction Score: <span style="font-size: 24px; font-weight: bold;">{score:.2f}</span></p>
            </div>
        """, unsafe_allow_html=True)

        # Explanation of the score and domain relevance
        st.markdown("""
            ### Understanding the Prediction Score:
            - A **higher score (> 0.5)** suggests that your slogan is **memorable, marketable, and impactful**.
            - A **lower score (< 0.5)** indicates that your slogan **might need refinement** to stand out better.

            ### Why Does This Matter?
            - A good slogan boosts **brand recognition** and **consumer recall**.
            - If multiple **domains point to your website**, it suggests strong online presence and SEO impact.
            - Avoiding **identical and similar trademarks** ensures **legal safety**.
            - An **available social media handle** makes marketing **easier and more effective**.
        """)

    else:
        st.error("❌ Please enter a slogan to analyze.")
