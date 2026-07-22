import json
import os

def load_students():
    try:
        with open("students.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []

def save_students(students):
    with open("students.json", "w") as file:
        json.dump(students, file, indent=4)

def add_student():
    students = load_students()
    
    while True:
        name = input("Enter Student Name: ").strip()
        if name:
            break
        print("Name cannot be empty!")
    
    while True:
        roll_number = input("Enter Roll Number: ").strip()
        if not roll_number:
            print("Roll Number cannot be empty!")
            continue
        
        found = False
        for student in students:
            if student["roll_number"] == roll_number:
                found = True
                break
        if found:
            print("Roll Number already exists! Enter another one.")
        else:
            break
    
    while True:
        try:
            age = int(input("Enter Age: "))
            if age > 0:
                break
            print("Age must be a positive number!")
        except ValueError:
            print("Please enter age in numbers only!")
    
    while True:
        course = input("Enter Course: ").strip()
        if course:
            break
        print("Course cannot be empty!")
    
    new_student = {
        "name": name,
        "roll_number": roll_number,
        "age": age,
        "course": course
    }
    
    students.append(new_student)
    save_students(students)
    print(f"\nStudent '{name}' added successfully!")

def display_students():
    students = load_students()
    if not students:
        print("\nNo students found in the system!")
        return
    
    print("\n" + "="*60)
    for idx, student in enumerate(students, 1):
        print(f"{idx}. Name: {student['name']}, Roll: {student['roll_number']}, Age: {student['age']}, Course: {student['course']}")
    print("="*60)
    print(f"Total Students: {len(students)}")

def search_student():
    roll_number = input("Enter Roll Number to Search: ").strip()
    students = load_students()
    
    for student in students:
        if student["roll_number"] == roll_number:
            print("\n" + "="*60)
            print(f"Student Found!")
            print(f"Name: {student['name']}")
            print(f"Roll Number: {student['roll_number']}")
            print(f"Age: {student['age']}")
            print(f"Course: {student['course']}")
            print("="*60)
            return
    
    print(f"\nStudent with Roll Number '{roll_number}' not found!")

def update_student():
    roll_number = input("Enter Roll Number to Update: ").strip()
    students = load_students()
    
    for student in students:
        if student["roll_number"] == roll_number:
            print("\n" + "="*60)
            print(f"Student Found!")
            print(f"Current Name: {student['name']}")
            print(f"Current Age: {student['age']}")
            print(f"Current Course: {student['course']}")
            print("="*60)
            
            while True:
                new_name = input("Enter New Name: ").strip()
                if new_name:
                    break
                print("Name cannot be empty!")
            
            while True:
                try:
                    new_age = int(input("Enter New Age: "))
                    if new_age > 0:
                        break
                    print("Age must be a positive number!")
                except ValueError:
                    print("Please enter age in numbers only!")
            
            while True:
                new_course = input("Enter New Course: ").strip()
                if new_course:
                    break
                print("Course cannot be empty!")
            
            student["name"] = new_name
            student["age"] = new_age
            student["course"] = new_course
            
            save_students(students)
            print(f"\nStudent with Roll Number '{roll_number}' updated successfully!")
            return
    
    print(f"\nStudent with Roll Number '{roll_number}' not found!")

def delete_student():
    roll_number = input("Enter Roll Number to Delete: ").strip()
    students = load_students()
    
    for student in students:
        if student["roll_number"] == roll_number:
            print("\n" + "="*60)
            print(f"Student Found!")
            print(f"Name: {student['name']}")
            print(f"Roll Number: {student['roll_number']}")
            print(f"Age: {student['age']}")
            print(f"Course: {student['course']}")
            print("="*60)
            
            confirm = input(f"\nAre you sure you want to delete this student? (yes/no): ").strip().lower()
            if confirm == "yes":
                students.remove(student)
                save_students(students)
                print(f"\nStudent with Roll Number '{roll_number}' deleted successfully!")
            else:
                print("\nDeletion cancelled!")
            return
    
    print(f"\nStudent with Roll Number '{roll_number}' not found!")

def main():
    print("\n===== Student Record Management System =====")
    print("Loading existing data...")
    students = load_students()
    if students:
        print(f"Loaded {len(students)} student(s) from file.")
    else:
        print("No existing data found. Starting fresh.")
    
    while True:
        print("\n===== Student Record Management System =====")
        print("1. Add Student")
        print("2. Display Students")
        print("3. Search Student")
        print("4. Update Student")
        print("5. Delete Student")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == "1":
            add_student()
        elif choice == "2":
            display_students()
        elif choice == "3":
            search_student()
        elif choice == "4":
            update_student()
        elif choice == "5":
            delete_student()
        elif choice == "6":
            print("\nExiting the program...")
            print("All data saved successfully!")
            break
        else:
            print("Invalid choice! Please enter a number between 1 and 6.")

if __name__ == "__main__":
    main()