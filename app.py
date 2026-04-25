import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("studydata2.csv")

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

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

st.title("🎓 Student Risk Prediction System")

h = st.number_input("Hours Studied", 0.0, 10.0)
att = st.number_input("Attendance %", 0.0, 100.0)
sleep = st.number_input("Sleep Hours", 0.0, 10.0)
marks = st.number_input("Internal Marks", 0.0, 100.0)
assign = st.number_input("Assignment Score", 0.0, 100.0)
backlogs = st.number_input("Backlogs", 0, 10)
stress = st.number_input("Stress Level", 0, 10)
internet = st.number_input("Internet Usage Hours", 0, 10)

if st.button("Predict Risk"):
    input_data = pd.DataFrame([[h, att, sleep, marks, assign, backlogs, stress, internet]],
    columns=X.columns)

    pred = model.predict(input_data)

    st.success(f"Predicted Risk Level: {pred[0]}")

    st.write("### Suggestions")

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