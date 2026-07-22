# Read and display file contents.
with open("students.txt", 'r') as file:
    contents = file.read()
    print(contents)

# Append new data to an existing file.
with open("students.txt", 'a') as file:
    name = input("Enter student name: ")
    age = input("Enter student age: ")
    major = input("Enter student major: ")
    file.write(f"\n{name}, {age}, {major}")

# Count the number of lines in a file.
with open("students.txt", 'r') as file:
    line_count = len(file.readlines())
    print(f"Number of lines in the file: {line_count}")