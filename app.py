import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# Use st.secrets for Streamlit Cloud, fall back to .env for local
api_key = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

st.set_page_config(page_title="Student Risk Predictor", page_icon="🎓", layout="wide")

data = pd.read_csv("studydata2.csv")
model = joblib.load("tempmodel.pkl")

# ---------------- Risk label fix ----------------
if data['risk_level'].nunique() < 3:
    def risk_category(row):
        if row["attendance_percentage"] < 60 or row["stress_level"] > 7:
            return 2
        elif row["attendance_percentage"] < 75 or row["sleep_hours"] < 5:
            return 1
        else:
            return 0

    data["risk_level"] = data.apply(risk_category, axis=1)

safe = data[data['risk_level'] == 0]
medium = data[data['risk_level'] == 1]
high = data[data['risk_level'] == 2]

if len(medium) > 0:
    medium = resample(medium, replace=True, n_samples=len(safe), random_state=42)
if len(high) > 0:
    high = resample(high, replace=True, n_samples=len(safe), random_state=42)

data = pd.concat([safe, medium, high])

# ---------------- ML training ----------------
X = data[[
    'hours_studied',
    'attendance_percentage',
    'sleep_hours',
    'internal_marks',
    'assignment_score',
    'backlogs',
    'stress_level',
    'internet_usage_hours'
]]

y = data['risk_level']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# ---------------- UI ----------------
st.title("🎓 Student Risk Prediction System")

name = st.text_input("Student Name")

st.sidebar.header("Enter Student Details")

h = st.sidebar.slider("Hours Studied", 0.0, 10.0, 2.0)
att = st.sidebar.slider("Attendance %", 0.0, 100.0, 75.0)
sleep = st.sidebar.slider("Sleep Hours", 0.0, 10.0, 6.0)
marks = st.sidebar.slider("Internal Marks", 0.0, 100.0, 50.0)
assign = st.sidebar.slider("Assignment Score", 0.0, 100.0, 50.0)
backlogs = st.sidebar.slider("Backlogs", 0, 10, 0)
stress = st.sidebar.slider("Stress Level", 0, 10, 5)
internet = st.sidebar.slider("Internet Usage Hours", 0, 10, 4)

# ---------------- Prediction ----------------
if st.button("🚀 Predict Risk"):

    input_data = pd.DataFrame([[h, att, sleep, marks, assign, backlogs, stress, internet]],
                              columns=X.columns)

    pred = model.predict(input_data)[0]
    
    # Store prediction in session state
    st.session_state.prediction = pred
    st.session_state.student_details = {
        "name": name,
        "hours_studied": h,
        "attendance": att,
        "sleep_hours": sleep,
        "internal_marks": marks,
        "assignment_score": assign,
        "backlogs": backlogs,
        "stress_level": stress,
        "internet_usage": internet
    }

    st.write(f"**Student Name:** {name}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Prediction Result")

        if pred == 2:
            st.error("🔴 HIGH RISK STUDENT")
        elif pred == 1:
            st.warning("🟡 MEDIUM RISK STUDENT")
        else:
            st.success("🟢 LOW RISK STUDENT")

    with col2:
        st.subheader("📌 Input Summary")
        st.write(f"Study Hours: {h}")
        st.write(f"Attendance: {att}%")
        st.write(f"Sleep Hours: {sleep}")
        st.write(f"Stress Level: {stress}")

    st.subheader("💡 Suggestions")

    if att < 75:
        st.warning("Improve attendance")
    if sleep < 5:
        st.warning("Increase sleep")
    if h < 2:
        st.warning("Increase study hours")
    if stress > 7:
        st.warning("Reduce stress")
    if backlogs > 0:
        st.warning("Clear backlogs")

# ---------------- AI CHATBOT ----------------
st.markdown("---")
st.subheader("🤖 AI Student Assistant (Groq)")

question = st.text_input("Ask AI about your result")

if question:
    if question.strip() == "":
        st.warning("Please enter a question")
        st.stop()

    # Build smart prompt with student details
    prompt = f"""
Student Details:
- Hours Studied: {st.session_state.student_details.get('hours_studied', h)}
- Classes Attended: {st.session_state.student_details.get('attendance', att)}%
- Sleep Hours: {st.session_state.student_details.get('sleep_hours', sleep)}
- Internal Marks: {st.session_state.student_details.get('internal_marks', marks)}
- Predicted Risk: {'HIGH RISK' if st.session_state.prediction == 2 else 'MEDIUM RISK' if st.session_state.prediction == 1 else 'LOW RISK'}

User Question: {question}

Explain the situation clearly and give suggestions to improve.
"""

    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant helping students understand their academic performance and risk. Give simple, clear advice."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    st.write(chat.choices[0].message.content)