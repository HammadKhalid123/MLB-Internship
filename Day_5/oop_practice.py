# Student Class

class Student:

    def __init__(self, name, roll_number, age, course):
        self.name = name
        self.roll_number = roll_number
        self.age = age
        self.course = course

    def display_info(self):
        print("\n----- Student Information -----")
        print("Name:", self.name)
        print("Roll Number:", self.roll_number)
        print("Age:", self.age)
        print("Course:", self.course)


student1 = Student("Hammad", "CS-101", 21, "Computer Science")
student2 = Student("Ali", "CS-102", 22, "Artificial Intelligence")

student1.display_info()
student2.display_info()

# Employee Class

class Employee:

    def __init__(self, name, employee_id, department, salary):
        self.name = name
        self.employee_id = employee_id
        self.department = department
        self.salary = salary

    def display_info(self):
        print("\n----- Employee Information -----")
        print("Name:", self.name)
        print("Employee ID:", self.employee_id)
        print("Department:", self.department)
        print("Salary:", self.salary)


employee1 = Employee("Ahmed", "EMP001", "Software Development", 85000)
employee2 = Employee("Sara", "EMP002", "Human Resources", 70000)

employee1.display_info()
employee2.display_info()

# Car Class

class Car:

    def __init__(self, brand, model, year, color):
        self.brand = brand
        self.model = model
        self.year = year
        self.color = color

    def start_car(self):
        print(f"{self.brand} {self.model} has started.")

    def stop_car(self):
        print(f"{self.brand} {self.model} has stopped.")

    def display_info(self):
        print("\n----- Car Information -----")
        print("Brand:", self.brand)
        print("Model:", self.model)
        print("Year:", self.year)
        print("Color:", self.color)


car1 = Car("Toyota", "Corolla", 2022, "White")
car2 = Car("Honda", "Civic", 2024, "Black")

car1.display_info()
car1.start_car()
car1.stop_car()

car2.display_info()
car2.start_car()
car2.stop_car()

# Inheritance Example

class Person:

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def display_info(self):
        print("\n----- Person Information -----")
        print("Name:", self.name)
        print("Age:", self.age)

    def introduce(self):
        print(f"My name is {self.name}.")


# Student Class

class Student(Person):

    def __init__(self, name, age, roll_number, course):
        super().__init__(name, age)
        self.roll_number = roll_number
        self.course = course

    def introduce(self):
        print(f"I am {self.name}. I study {self.course}.")

    def display_student(self):
        print("\n----- Student Information -----")
        print("Name:", self.name)
        print("Age:", self.age)
        print("Roll Number:", self.roll_number)
        print("Course:", self.course)

# Teacher Class

class Teacher(Person):

    def __init__(self, name, age, subject, experience):
        super().__init__(name, age)
        self.subject = subject
        self.experience = experience

    def introduce(self):
        print(f"I am {self.name}. I teach {self.subject}.")

    def display_teacher(self):
        print("\n----- Teacher Information -----")
        print("Name:", self.name)
        print("Age:", self.age)
        print("Subject:", self.subject)
        print("Experience:", self.experience, "years")


# Creating Objects

student1 = Student("Hammad", 21, "CS-101", "Computer Science")
teacher1 = Teacher("Ahmed", 35, "Python Programming", 8)

# Accessing Inherited Methods

student1.display_student()
student1.introduce()

teacher1.display_teacher()
teacher1.introduce()

# Accessing Inherited Attributes

print("\n----- Accessing Inherited Attributes -----")
print("Student Name:", student1.name)
print("Student Age:", student1.age)

print("Teacher Name:", teacher1.name)
print("Teacher Age:", teacher1.age)