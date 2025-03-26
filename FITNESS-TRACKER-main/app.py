import streamlit as st
import numpy as np
import pandas as pd
import time
import warnings
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# Set Streamlit page config
st.set_page_config(page_title="Personal Fitness Tracker", layout="wide")

# Title and introduction
st.markdown("<h1 style='text-align: center;'>Personal Fitness Tracker</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict your burned calories and get AI-powered fitness advice.</p>", unsafe_allow_html=True)

st.divider()

# Sidebar: User input
st.sidebar.header("Enter Your Details")

def user_input_features():
    age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=30)
    bmi = st.sidebar.number_input("BMI", min_value=15.0, max_value=40.0, value=20.0, format="%.1f")
    duration = st.sidebar.number_input("Duration (min)", min_value=0, max_value=60, value=15)
    heart_rate = st.sidebar.number_input("Heart Rate (bpm)", min_value=50, max_value=180, value=80)
    body_temp = st.sidebar.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, format="%.1f")
    gender_button = st.sidebar.radio("Gender", ("Male", "Female"))

    gender = 1 if gender_button == "Male" else 0

    # Create DataFrame
    features = pd.DataFrame({
        "Age": [age], "BMI": [bmi], "Duration": [duration],
        "Heart_Rate": [heart_rate], "Body_Temp": [body_temp], "Gender_male": [gender]
    })
    
    return features

df = user_input_features()

# Display user parameters
st.subheader("Your Input Data")
st.dataframe(df, use_container_width=True)

st.divider()

# Load and preprocess data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

# Merge and clean data
exercise_df = exercise.merge(calories, on="User_ID").drop(columns="User_ID")
exercise_df["BMI"] = round(exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2), 2)

# Prepare training data
exercise_train, exercise_test = train_test_split(exercise_df, test_size=0.2, random_state=1)
features_cols = ["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp"]
exercise_train = pd.get_dummies(exercise_train[features_cols + ["Calories"]], drop_first=True)
exercise_test = pd.get_dummies(exercise_test[features_cols + ["Calories"]], drop_first=True)

# Train model
X_train, y_train = exercise_train.drop("Calories", axis=1), exercise_train["Calories"]
model = RandomForestRegressor(n_estimators=1000, max_depth=6, max_features=3)
model.fit(X_train, y_train)

# Predict calories burned
df = df.reindex(columns=X_train.columns, fill_value=0)
prediction = model.predict(df)

st.subheader("Calories Burned Prediction")
st.success(f"Estimated Calories Burned: {round(prediction[0], 2)} kcal")

st.divider()

# Insights Section
st.subheader("Your Fitness Insights")

col1, col2, col3, col4 = st.columns(4)

calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]

# Comparison stats
def calc_percentage(feature, value):
    return round((exercise_df[feature] < value).sum() / len(exercise_df) * 100, 2)

col1.metric(label="Older Than", value=f"{calc_percentage('Age', df['Age'][0])}%", delta="Compared to users")
col2.metric(label="Longer Duration", value=f"{calc_percentage('Duration', df['Duration'][0])}%", delta="Compared to users")
col3.metric(label="Higher Heart Rate", value=f"{calc_percentage('Heart_Rate', df['Heart_Rate'][0])}%", delta="Compared to users")
col4.metric(label="Higher Body Temp", value=f"{calc_percentage('Body_Temp', df['Body_Temp'][0])}%", delta="Compared to users")

st.divider()

# Show similar results
st.subheader("Similar Results")
st.dataframe(similar_data.sample(min(5, len(similar_data))), use_container_width=True)

st.divider()

# === Groq AI Integration for Fitness Advice ===
GROQ_API_KEY = "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"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

def get_fitness_advice(user_query):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",  # Choose an appropriate model
        "messages": [
            {"role": "system", "content": "You are an expert fitness coach."},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(GROQ_ENDPOINT, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        logger.info(f"API response status code: {response.status_code}")
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as http_err:
        error_message = f"HTTP error occurred: {http_err}"
        logger.error(error_message)
        logger.error(f"Status code: {http_err.response.status_code}")
        logger.error(f"Response text: {http_err.response.text}")
        return f"Error: Unable to fetch AI response due to HTTP error ({http_err.response.status_code}). Response: {http_err.response.text}"
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Connection error occurred: {conn_err}")
        return "Error: Unable to fetch AI response due to connection error."
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"Timeout error occurred: {timeout_err}")
        return "Error: Unable to fetch AI response due to timeout error."
    except Exception as err:
        logger.error(f"An error occurred: {err}")
        return "Error: Unable to fetch AI response due to unspecified error."

# User input for AI Fitness Coach
st.subheader("💡 AI Fitness Coach (Powered by Groq)")
user_query = st.text_area("Ask AI for fitness advice (e.g., best exercises, diet tips, recovery methods)")

if st.button("Get Advice"):
    with st.spinner("Fetching AI-powered advice..."):
        advice = get_fitness_advice(user_query)
        time.sleep(2)
    st.write("### AI Advice:")
    st.write(advice)

