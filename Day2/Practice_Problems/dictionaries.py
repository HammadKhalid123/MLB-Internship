
# Question 1: Create a Student Record Dictionary

student = {
    "name": "Hadeed",
    "roll_number": "CS-101",
    "age": 21,
    "course": "Computer Science"
}

print("=" * 50)
print("Question 1: Create a Student Record Dictionary")
print("=" * 50)

print("Student Record:")
for key, value in student.items():
    print(f"{key}: {value}")

# Question 2: Calculate Average Marks of Students

marks = {
    "Math": 90,
    "English": 85,
    "Physics": 88,
    "Computer": 95
}

total_marks = 0

for mark in marks.values():
    total_marks += mark

average = total_marks / len(marks)

print("\n" + "=" * 50)
print("Question 2: Calculate Average Marks")
print("=" * 50)

print("Marks:", marks)
print("Average Marks:", round(average, 2))

# Question 3: Count Frequency of Words in a Sentence

sentence = input("\nEnter a sentence: ")

words = sentence.lower().split()

word_frequency = {}

for word in words:
    if word in word_frequency:
        word_frequency[word] += 1
    else:
        word_frequency[word] = 1

print("\n" + "=" * 50)
print("Question 3: Count Frequency of Words")
print("=" * 50)

for word, count in word_frequency.items():
    print(f"{word} : {count}")