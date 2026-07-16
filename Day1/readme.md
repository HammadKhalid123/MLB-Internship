# Day 1 - Student Grading System

## Objective

Build a Student Grading System using Python while practicing Python fundamentals and Git/GitHub workflow.
---

## Features

- Student Name Input
- Student Class Input
- Multiple Subjects Input
- Marks Validation (0–100)
- Total Marks Calculation
- Average Marks Calculation
- Grade Assignment (A, B, C, F)
- Pass/Fail Status
- Formatted Student Report

---

## Concepts Covered

### Python

- Variables
- Data Types
- Lists
- Functions
- Loops (`for` & `while`)
- Conditional Statements (`if`, `elif`, `else`)
- Input Validation

### Git & GitHub

- Creating a Repository
- Git Initialization (`git init`)
- Git Status (`git status`)
- Staging Changes (`git add`)
- Creating Commits (`git commit`)
- Pushing Code to GitHub (`git push`)

---

## Example Scenario

**Input**

```text
Enter Student Name: Hammad
Enter Class: BSCS

Enter Number of Subjects: 3

Subject 1: Python
Marks: 90

Subject 2: AI
Marks: 85

Subject 3: Math
Marks: 95
```

**Output**

```text
=============== STUDENT REPORT ===============

Name    : Hammad
Class   : BSCS

Subjects & Marks
---------------------------------------------
Python               90
AI                   85
Math                 95
---------------------------------------------
Total Marks : 270
Average     : 90.00
Grade       : A
Status      : PASS

==============================================
```

---

## Project Structure

```text
Day1/
├── grading_system.py
└── README.md
```