# Day 4 - File Handling & JSON in Python

## Overview

Today, I learned how to work with files and JSON in Python. I practiced reading, writing, and updating files, and built a persistent Student Record Management System where student records are permanently stored in a JSON file.

---

# Topics Covered

- File Handling in Python
- Reading Files
- Writing Files
- Appending Data
- File Modes (r, w, a)
- Using the `with` Statement
- JSON in Python
- Reading JSON Files
- Writing JSON Files
- Converting Python Dictionaries to JSON
- Loading JSON Data into Python Objects
- Exception Handling

---

# Practice Programs

## File Handling

Implemented the following programs:

- Create a text file and write data into it
- Read and display file contents
- Append new data to an existing file
- Count the number of lines in a file

## JSON

Implemented the following programs:

- Store student information in a JSON file
- Read data from a JSON file
- Update an existing student's information
- Add a new student to the JSON file

---

# Student Record Management System (Persistent Version)

## Features

The application includes the following features:

- Add Student
- View All Students
- Search Student by Roll Number
- Update Student Information
- Delete Student
- Automatically Load Student Records from a JSON File
- Automatically Save Changes to the JSON File
- Input Validation
- Exception Handling

---

# Data Storage

Student records are stored permanently in a JSON file.

Each student record contains:

- Name
- Roll Number
- Age
- Course

---

# What I Learned

- How to work with text files in Python.
- How to read, write, and append data to files.
- How JSON is used to store structured data.
- How to convert Python dictionaries into JSON format.
- How to load JSON data back into Python objects.
- How to build applications that permanently store data.

---

# How File Handling and JSON Work Together

File handling is used to create, read, update, and save files on the computer.

JSON provides a structured format for storing data, making it easy to save Python dictionaries and lists. By combining file handling with JSON, applications can permanently store and retrieve information even after the program is closed.

---

# Challenges Faced

- Understanding how JSON stores Python data.
- Updating existing records without losing previous data.
- Managing file reading and writing correctly.
- Handling invalid user input using exception handling.

---

# Solutions Implemented

- Used the `json` module to read and write JSON data.
- Used the `with` statement for safe file handling.
- Added input validation to prevent invalid data.
- Saved every change immediately to ensure data persistence.

---

# Files Included

- Practice Programs
- Student Record Management System
- Sample JSON File
- README.md

---

# Conclusion

Day 4 helped me understand how Python applications can permanently store and manage data using file handling and JSON.

I also learned how real-world applications save, retrieve, and update information efficiently, making this an important step toward building larger software projects.