import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

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

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

print("Model Accuracy:", model.score(x_test, y_test))

joblib.dump(model, "tempmodel.pkl")

print("\n--- Student Risk Prediction ---")

name = input("Student Name: ")
h = float(input("Hours studied: "))
att = float(input("Attendance %: "))
sleep = float(input("Sleep hours: "))
marks = int(input("Internal marks: "))
assign = int(input("Assignment score: "))
backlogs = int(input("Backlogs: "))
stress = int(input("Stress level: "))
internet = int(input("Internet usage hours: "))

new_data = pd.DataFrame([[h, att, sleep, marks, assign, backlogs, stress, internet]],
columns=[
    'hours_studied',
    'attendance_percentage',
    'sleep_hours',
    'internal_marks',
    'assignment_score',
    'backlogs',
    'stress_level',
    'internet_usage_hours'
])

pred = model.predict(new_data)

print(f"\nStudent Name: {name}")
print("\nPredicted Risk Level:", pred[0])

print("\n--- Suggestions ---")

if att < 75:
    print("Improve attendance")
if sleep < 5:
    print("Increase sleep")
if h < 2:
    print("Increase study hours")
if stress > 7:
    print("Reduce stress")
if backlogs > 0:
    print("Clear backlogs")