# Number Analysis Tool


# Function to check even or odd
def check_even_odd(number):
    if number % 2 == 0:
        return "Even"
    else:
        return "Odd"


# Function to check prime number
def check_prime(number):
    if number <= 1:
        return False

    for i in range(2, number):
        if number % i == 0:
            return False

    return True


# Function to count digits
def count_digits(number):
    if number == 0:
        return 1

    count = 0
    number = abs(number)

    while number > 0:
        number = number // 10
        count += 1

    return count


# Function to reverse number
def reverse_number(number):
    reverse = 0
    temp = abs(number)

    while temp > 0:
        remainder = temp % 10
        reverse = (reverse * 10) + remainder
        temp = temp // 10

    if number < 0:
        reverse = -reverse

    return reverse


# Function to check palindrome
def check_palindrome(number):
    if number < 0:
        return False

    reverse = reverse_number(number)

    if number == reverse:
        return True
    else:
        return False


# ---------------- Main Program ----------------

number = int(input("Enter a number: "))

print("\n========== Number Analysis ==========")

print("Number:", number)

# Even or Odd
print("Even/Odd:", check_even_odd(number))

# Prime
if check_prime(number):
    print("Prime Number: Yes")
else:
    print("Prime Number: No")

# Digits
print("Total Digits:", count_digits(number))

# Reverse
print("Reversed Number:", reverse_number(number))

# Palindrome
if check_palindrome(number):
    print("Palindrome: Yes")
else:
    print("Palindrome: No")

print("=====================================")