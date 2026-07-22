# Student Information Management System - Practice Tasks
import json

# TASK 1: Store student information in a JSON file
students = [
    {"name": "John Doe", "age": 21, "roll_number": 102, "major": "Physics"},
    {"name": "Jane Smith", "age": 22, "roll_number": 103, "major": "Chemistry"},
    {"name": "Alice Johnson", "age": 20, "roll_number": 104, "major": "Computer Science"}
]

with open("students.json", "w") as file:
    json.dump(students, file, indent=4) 

print("Students data saved successfully!")
print("-" * 50)

# TASK 2: Read data from a JSON file

with open("students.json", "r") as file:
    data = json.load(file)

print("\n All Students Data:")
for student in data:
    print(f"  Name: {student['name']}, Roll: {student['roll_number']}, "
          f"Age: {student['age']}, Major: {student['major']}")


# TASK 3: Update an existing student's information

print("\n Update Student Information")
roll_input = int(input("Enter student roll number to update: "))

with open("students.json", "r") as file:
    students_data = json.load(file)

found = False
for student in students_data:
    if student['roll_number'] == roll_input:
        print(f"\n Student found: {student['name']}")
        
        new_age = int(input(f"Enter new age (current: {student['age']}): "))
        new_major = input(f"Enter new major (current: {student['major']}): ")
        student['age'] = new_age
        student['major'] = new_major
        found = True
        print(f"\n Student updated successfully!")
        print(f"Updated: {student['name']} - Age: {student['age']}, Major: {student['major']}")
        break

if not found:
    print(f"Student with roll number {roll_input} not found!")

with open("students.json", "w") as file:
    json.dump(students_data, file, indent=4)


# TASK 4: Add a new student to the JSON file

print("\n Add New Student")
print("Enter new student details:")

new_name = input("Name: ")
new_age = int(input("Age: "))
new_roll = int(input("Roll Number: "))
new_major = input("Major: ")

new_student = {
    "name": new_name,
    "age": new_age,
    "roll_number": new_roll,
    "major": new_major
}

with open("students.json", "r") as file:
    students_data = json.load(file)
roll_exists = any(student['roll_number'] == new_roll for student in students_data)

if roll_exists:
    print(f"Roll number {new_roll} already exists! Cannot add duplicate.")
else:
    students_data.append(new_student)
    with open("students.json", "w") as file:
        json.dump(students_data, file, indent=4)
    
    print(f"\nNew student added successfully!")
    print(f"   {new_name} - Roll: {new_roll}, Age: {new_age}, Major: {new_major}")

# FINAL: Display all students after all operations

print("\nFinal Students List:")
with open("students.json", "r") as file:
    final_data = json.load(file)

for idx, student in enumerate(final_data, 1):
    print(f"  {idx}. {student['name']} (Roll: {student['roll_number']}) - "
          f"Age: {student['age']}, Major: {student['major']}")

print("\nAll tasks completed successfully!")