# Menu Driven Application


# Function to Check Prime Number
def check_prime():
    number = int(input("Enter a number: "))

    if number <= 1:
        print(number, "is not a Prime Number.")
    else:
        for i in range(2, number):
            if number % i == 0:
                print(number, "is not a Prime Number.")
                break
        else:
            print(number, "is a Prime Number.")


# Function to Generate Fibonacci Series
def fibonacci():
    n_terms = int(input("Enter the number of terms: "))

    n1 = 0
    n2 = 1

    print("Fibonacci Series:")

    for i in range(n_terms):
        print(n1)
        nth = n1 + n2
        n1 = n2
        n2 = nth


# Function to Check Palindrome
def palindrome():
    number = int(input("Enter a number: "))

    original_number = number
    reverse = 0

    while number > 0:
        remainder = number % 10
        reverse = (reverse * 10) + remainder
        number = number // 10

    if original_number == reverse:
        print(original_number, "is a Palindrome.")
    else:
        print(original_number, "is not a Palindrome.")


# Function to Print Multiplication Table
def multiplication_table():
    number = int(input("Enter a number: "))

    print("Multiplication Table of", number)

    for i in range(1, 11):
        print(number, "x", i, "=", number * i)


# ---------------- Main Program ----------------

while True:

    print("\n========== Menu ==========")
    print("1. Check Prime Number")
    print("2. Generate Fibonacci Series")
    print("3. Check Palindrome")
    print("4. Generate Multiplication Table")
    print("5. Exit")
    print("==========================")

    choice = int(input("Enter your choice: "))

    if choice == 1:
        print("\n----- Prime Number Check -----")
        check_prime()

    elif choice == 2:
        print("\n----- Fibonacci Series -----")
        fibonacci()

    elif choice == 3:
        print("\n----- Palindrome Check -----")
        palindrome()

    elif choice == 4:
        print("\n----- Multiplication Table -----")
        multiplication_table()

    elif choice == 5:
        print("\nThank you for using the program.")
        print("Goodbye!")
        break

    else:
        print("Invalid Choice! Please enter a number between 1 and 5.")