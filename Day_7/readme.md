# Day 7 – Data Cleaning & Visualization

## Overview

This project focuses on preparing a student performance dataset for analysis and presenting meaningful insights through data visualization. Using Pandas, I cleaned the dataset by handling common preprocessing tasks, then used Matplotlib and Seaborn to create different charts. Finally, I built a simple Student Performance Dashboard to summarize key statistics and highlight important trends.

---

## Topics Covered

### Data Cleaning (Pandas)

- Reading and exploring CSV data
- Identifying missing values
- Removing duplicate records
- Renaming columns
- Creating calculated columns
- Sorting and filtering data
- Exporting the cleaned dataset

### Data Visualization

- Bar Chart
- Histogram
- Scatter Plot
- Pie Chart
- Box Plot

---

## Files Included

- `Data_Cleaning/data_cleaning.ipynb`
- `Data_Visualization/Data_Visualization.ipynb`
- `Data_Visualization/Student Performance Dashboard.ipynb`
- `Data_Cleaning/student_performance.csv`
- `/cleaned_student_performance.csv`
- `Data_Visualization/Charts/`
  - `bar_chart.png`
  - `histogram.png`
  - `scatter_plot.png`
  - `pie_chart.png`
  - `count_plot.png`

---

## Data Cleaning Steps

- Loaded the student performance dataset
- Inspected the dataset for missing values
- Removed duplicate entries
- Renamed columns for better readability
- Calculated each student's `Average_Score`
- Assigned performance categories based on average score
- Sorted the dataset by performance
- Exported the cleaned dataset for visualization

---

## Visualizations Created

- **Bar Chart:** Average score of each student
- **Histogram:** Distribution of average scores
- **Scatter Plot:** Relationship between Python and Machine Learning marks
- **Pie Chart:** Distribution of performance categories
- **Box Plot:** Comparison of marks across all subjects

---

## Key Insights

- Most students fall into the **Good** and **Average** performance categories.
- Students with higher Python marks generally achieved higher Machine Learning scores.
- The dashboard provides a quick overview of top-performing students, subject averages, and students who need improvement.

---

## Challenges Faced

- Organizing the data cleaning workflow before visualization.
- Choosing suitable charts for different types of analysis.
- Maintaining a consistent dataset throughout the project.

---

## Lessons Learned

- Clean and well-structured data improves the quality of analysis.
- Different visualization techniques reveal different patterns in the data.
- Combining Pandas, Matplotlib, and Seaborn makes it easier to analyze and present data effectively.