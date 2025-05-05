import streamlit as st
import re
import pickle
import base64
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline

# ==================== Load Pretrained Models ====================

# Load Random Forest model
with open('model_rf.pickle', 'rb') as f:
    rf_model = pickle.load(f)

# Load MLP model
with open('model_mlp.pickle', 'rb') as f:
    mlp_model = pickle.load(f)

# Load CountVectorizer
with open('vector.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

# Load GPT Sentiment Analyzer (BERT-based)
sentiment_analyzer = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

# ==================== Helper Functions ====================

# Improved Text Cleaning Function (Keeps Emotional Words)
def cleanup(sentence):
    return re.sub(r"[^a-zA-Z' ]+", '', str(sentence)).strip().lower()

# Improved GPT Sentiment Analysis Function
def gpt_sentiment_analysis(text):
    result = sentiment_analyzer(text)
    label = int(result[0]['label'].split()[0])  # Extract numerical rating

    if label >= 4:
        return "Positive"
    elif label == 3:
        return "Neutral"
    else:
        return "Negative"

# Predict Sentiment Function
def predict_sentiment(user_input):
    cleaned_text = cleanup(user_input)

    # Vectorization
    vectorized_text = vectorizer.transform([cleaned_text])

    # Model Predictions
    rf_pred = rf_model.predict(vectorized_text)[0]
    mlp_pred = mlp_model.predict(vectorized_text)[0]
    gpt_pred = gpt_sentiment_analysis(cleaned_text)

    # Convert predictions to text
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    rf_sentiment = sentiment_map[rf_pred]
    mlp_sentiment = sentiment_map[mlp_pred]

    # Aggregation with GPT Priority
    sentiment_votes = [rf_sentiment, mlp_sentiment, gpt_pred]

    # Prioritize GPT prediction if it's "Positive"
    if gpt_pred == "Positive":
        final_sentiment = "Positive"
    else:
        final_sentiment = max(set(sentiment_votes), key=sentiment_votes.count)  # Most common prediction

    return final_sentiment

# ==================== Streamlit UI Enhancements ====================

# Background Image Function
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{encoded_string});
        background-size: cover;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Call background function
add_bg_from_local('5.jpeg')  # Ensure you have '1.png' in the same directory

# ==================== Streamlit Page Layout ====================

# Page Title with Styling
st.markdown(
    """
    <h1 style="text-align: center; color: black; font-size: 40px; font-family: Caveat, sans-serif;">
        Mining the Opinions of Software Developers
    </h1>
    <h2 style="text-align: center; color: black ;">For Improved Project Insights</h2>
    <hr style="border: 1px solid black;">
    """,
    unsafe_allow_html=True
)

# User Input
user_input = st.text_input("Enter Text for Sentiment Analysis:")

# Predict Button
if st.button("PREDICT"):
    if user_input.strip():  # This ensures input is not empty or just spaces
        final_result = predict_sentiment(user_input)

        # Display Final Conclusion (NO Background Color)
        st.markdown(f"""
        <h2 style="text-align: center; color: black;">Predicted OPINION = {final_result.upper()}</h2>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Please enter a valid text input.")
