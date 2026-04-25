import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("studydata2.csv")

# 1. Risk Level Distribution
plt.figure()
sns.countplot(x="risk_level", data=data)
plt.title("Risk Level Distribution")
plt.show()

# 2. Attendance vs Risk
plt.figure()
sns.boxplot(x="risk_level", y="attendance_percentage", data=data)
plt.title("Attendance vs Risk Level")
plt.show()

# 3. Study Hours vs Risk
plt.figure()
sns.boxplot(x="risk_level", y="hours_studied", data=data)
plt.title("Study Hours vs Risk Level")
plt.show()

# 4. Correlation Heatmap
plt.figure()
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()