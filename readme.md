# 🎓 Student Attendance Risk Prediction System

## 📌 Project Overview

This project is a Machine Learning-based system that predicts whether a student is at risk based on academic and behavioral factors such as study hours, attendance percentage, sleep hours, internal marks, and other performance indicators. It helps identify students who may need academic support early.

---

## 🚀 Features

* Predicts student risk level (Low / Medium / High)
* Uses multiple student performance features
* Simple CLI-based input system
* Provides improvement suggestions
* Built using Machine Learning (Logistic Regression)

---

## 📊 Dataset Information

The dataset contains the following features:

* hours_studied
* attendance_percentage
* sleep_hours
* internal_marks
* assignment_score
* backlogs
* stress_level
* internet_usage_hours
* final_result
* risk_level

---

## 🧠 Machine Learning Model

* Algorithm: Logistic Regression
* Library: Scikit-learn
* Task: Classification

---

## 📂 Project Structure

```
attendance-risk-project/
│
├── main.py              # Main prediction script
├── studydata.csv        # Dataset file
├── studydata2.csv       # Dataset file 2
├── visualization.py     # Data visualization (optional)
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/project-name.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
python main.py
```

---

## ▶️ How it Works

1. User enters student details (study hours, attendance, etc.)
2. Model processes the input
3. System predicts risk level
4. Suggestions are shown based on performance

---

## 📈 Example Output

```
Model Accuracy: 0.87

--- Student Risk Prediction ---
Predicted Risk Level: High

--- Suggestions ---
Improve attendance
Increase sleep
Reduce stress
```

---

## 🎯 Future Improvements

* Flask web application
* Graphical dashboard
* Advanced ML models (Random Forest / XGBoost)
* Cloud deployment
* Real-time student tracking system

---

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* NumPy

---

## 👨‍💻 Author

This project is developed for academic ML learning and student performance analysis.

---

## ⭐ Outcome

Helps in early identification of students at risk and supports academic intervention strategies.