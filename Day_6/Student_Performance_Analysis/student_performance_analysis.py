import pandas as pd

# ---------------------------------------
# Load the dataset
# ---------------------------------------
df = pd.read_csv("student_performance.csv")

# ---------------------------------------
# Display basic information
# ---------------------------------------
print("===== Dataset Information =====")
df.info()

# ---------------------------------------
# Subject columns
# ---------------------------------------
mark_subjects = [
    "Mathematics",
    "Statistics",
    "Python",
    "Machine_Learning"
]

# ---------------------------------------
# Calculate average marks for each subject
# ---------------------------------------
print("\n===== Average Marks of Each Subject =====")

for subject in mark_subjects:
    avg = df[subject].mean()
    print(f"{subject}: {avg:.2f}")

attendance_avg = df["Attendance"].mean()
print(f"Attendance: {attendance_avg:.2f}")

# ---------------------------------------
# Add Total and Average Marks columns
# ---------------------------------------
df["Total"] = df[mark_subjects].sum(axis=1)

df["Average_Marks"] = df[mark_subjects].mean(axis=1)

# ---------------------------------------
# Find overall average
# ---------------------------------------
overall_average = df["Total"].mean()

print(f"\nOverall Average Total Marks: {overall_average:.2f}")

# ---------------------------------------
# Mark students as Above/Below Average
# ---------------------------------------
df["Performance_Status"] = df["Total"].apply(
    lambda x: "Above Average" if x >= overall_average else "Below Average"
)

# ---------------------------------------
# Identify Top 5 Students
# ---------------------------------------
top5_students = df.nlargest(5, "Total")

print("\n===== Top 5 Performing Students =====")
print(top5_students)

# Add Top 5 column
df["Top_5_Performer"] = "No"
df.loc[top5_students.index, "Top_5_Performer"] = "Yes"

# ---------------------------------------
# Display students below average
# ---------------------------------------
print("\n===== Students Scoring Below Average =====")

below_average = df[df["Performance_Status"] == "Below Average"]
print(below_average)

# ---------------------------------------
# Display total number of students
# ---------------------------------------
print(f"\nTotal Number of Students: {len(df)}")

# ---------------------------------------
# Save analyzed dataset
# ---------------------------------------
df.to_csv("processed_students_performance.csv", index=False)

print("\nAnalyzed dataset saved successfully as 'processed_students.csv'")