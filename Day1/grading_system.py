def get_student_info():
    name = input("Enter Student Name: ")
    student_class = input("Enter Class: ")
    return name, student_class

def get_subjects():
    total_subjects = int(input("Enter Number of Subjects: "))
    subjects = []
    marks = []

    for i in range(total_subjects):
        subject = input(f"\nEnter Subject {i + 1} Name: ")
        subjects.append(subject)

        while True:
            mark = float(input(f"Enter Marks for {subject} (0-100): "))
            if 0 <= mark <= 100:
                marks.append(mark)
                break
            else:
                print("Invalid Marks! Please enter marks between 0 and 100.")

    return subjects, marks


def calculate_average(marks):
    total = sum(marks)
    average = total / len(marks)
    return total, average

def calculate_grade(average):
    if average >= 90:
        return "A"
    elif average >= 80:
        return "B"
    elif average >= 70:
        return "C"
    else:
        return "F"

def calculate_status(average):
    if average >= 50:
        return "PASS"
    return "FAIL"


def display_report(name, student_class, subjects, marks, total, average, grade, status):
    print("==========================================")
    print("           STUDENT REPORT")
    print("==========================================")

    print(f"Name : {name}")
    print(f"Class : {student_class}")
    print("\nSubjects & Marks")
    print("==========================================")

    for i in range(len(subjects)):
        print(f"{subjects[i]} : {marks[i]}")

    print("==========================================")
    print(f"Total Marks : {total}")
    print(f"Average : {average:.2f}")
    print(f"Grade : {grade}")
    print(f"Status : {status}")
    print("==========================================")


def main():
    name, student_class = get_student_info()
    subjects, marks = get_subjects()
    total, average = calculate_average(marks)

    grade = calculate_grade(average)
    status = calculate_status(average)

    display_report(name,student_class,subjects,marks,total,average,grade,status)

main()