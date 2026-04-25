import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("studydata.csv")

data["AttendancePercent"] = (data["ClassesAttended"] / data["TotalClasses"]) * 100
X = data[['HoursStudied','ClassesAttended','TotalClasses','SleepHours','InternalMarks']]
y = data['LowAttendance']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(x_train, y_train)

h = float(input("Hours studied (1 to 5): "))
att = int(input("Classes attended (1 to 35): "))
total = int(input("Total classes (1 to 30): "))
sleep = float(input("Sleep hours (1 to 8): "))
marks = int(input("Internal marks (1 to 100): "))

new_data = pd.DataFrame([[h, att, total, sleep, marks]],
columns=['HoursStudied','ClassesAttended','TotalClasses','SleepHours','InternalMarks'])
features = ['HoursStudied','ClassesAttended','TotalClasses','SleepHours','InternalMarks']

pred = model.predict(new_data)

if pred[0] == 1:
    print("\n ⚠️ Student at risk (low attendance)\n")
else:
    print("\n ✅ Attendance is safe\n")