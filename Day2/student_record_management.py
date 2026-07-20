students_list = []


def main():

    while True:
        st_name = input("Enter Student Name: ")
        if st_name:
            break
        else:
            print("Name cannot be empty!")

    while True:
        roll_number = input("Enter Roll Number: ")
        found = False

        for student in students_list:
            if student["roll_number"] == roll_number:
                found = True
                break
        if found:
            print("Roll Number already exists. Enter another one.")
        else:
            break

    while True:

        try:
            age = int(input("Enter Age: "))
            break
        except:
            print("Please enter age in numbers only.")

    while True:

        course = input("Enter Course: ")
        if course:
            break
        else:
            print("Course cannot be empty!")

    return st_name, roll_number, age, course


def add_student(st_name, roll_number, age, course):

    student_dict = {
        "name": st_name,
        "roll_number": roll_number,
        "age": age,
        "course": course
    }

    students_list.append(student_dict)
    print(f"\nStudent '{st_name}' added successfully!")


def display_students():

    if not students_list:
        print("\nNo students found.")
        return
    print("\n===== Student List =====")

    for student in students_list:

        print("Name        :", student["name"])
        print("Roll Number :", student["roll_number"])
        print("Age         :", student["age"])
        print("Course      :", student["course"])
        print("-" * 30)

    print("Total Students:", len(students_list))

def search_student(roll_number):

    for student in students_list:
        if student["roll_number"] == roll_number:
            print("\nStudent Found")

            print("Name        :", student["name"])
            print("Roll Number :", student["roll_number"])
            print("Age         :", student["age"])
            print("Course      :", student["course"])

            return
    print("Student not found.")



def update_student(roll_number):

    for student in students_list:

        if student["roll_number"] == roll_number:
            student["name"] = input("Enter New Name: ")

            while True:
                try:
                    student["age"] = int(input("Enter New Age: "))
                    break
                except:
                    print("Enter age in numbers only.")
            student["course"] = input("Enter New Course: ")
            print("Student updated successfully!")
            return
        
    print("Student not found.")



def delete_student(roll_number):
    for student in students_list:
        if student["roll_number"] == roll_number:
            students_list.remove(student)
            print("Student deleted successfully!")

            return
    print("Student not found.")

# Main Program

if __name__ == "__main__":
    while True:
        print("\n===== Student Record Management System =====")

        print("1. Add Student")
        print("2. Display Students")
        print("3. Search Student")
        print("4. Update Student")
        print("5. Delete Student")
        print("6. Exit")

        choice = input("Enter your choice (1-6): ")

        if choice == "1":
            st_name, roll_number, age, course = main()
            add_student(st_name, roll_number, age, course)

        elif choice == "2":
            display_students()

        elif choice == "3":
            roll_number = input("Enter Roll Number to Search: ")
            search_student(roll_number)

        elif choice == "4":
            roll_number = input("Enter Roll Number to Update: ")
            update_student(roll_number)

        elif choice == "5":
            roll_number = input("Enter Roll Number to Delete: ")
            delete_student(roll_number)

        elif choice == "6":
            print("Exiting the program...")
            break

        else:
            print("Invalid choice! Please try again.")