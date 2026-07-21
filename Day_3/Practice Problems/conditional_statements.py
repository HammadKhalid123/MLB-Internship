# Q: Check whether a number is positive, negative, or zero.

print("========== Positive, Negative or Zero ==========")

number = int(input("Enter a number: "))

if number > 0:
    print("Result: The number is Positive.")
elif number < 0:
    print("Result: The number is Negative.")
else:
    print("Result: The number is Zero.")

# Q: Check whether a number is even or odd.

print("\n=============== Even or Odd ====================")

number = int(input("Enter a number: "))

if number % 2 == 0:
    print("Result: The number is Even.")
else:
    print("Result: The number is Odd.")

# Q: Find the largest among three numbers.

print("\n========== Largest Among Three Numbers =========")

num1 = int(input("Enter first number: "))
num2 = int(input("Enter second number: "))
num3 = int(input("Enter third number: "))

if (num1 >= num2) and (num1 >= num3):
    largest = num1
elif (num2 >= num1) and (num2 >= num3):
    largest = num2
else:
    largest = num3

print("Largest Number:", largest)

# Q: Check whether a year is a leap year or not.

print("\n================ Leap Year Check ===============")

year = int(input("Enter a year: "))

if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
    print(year, "is a Leap Year.")
else:
    print(year, "is not a Leap Year.")

print("===============================================")