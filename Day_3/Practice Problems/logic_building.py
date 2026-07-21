# Q: Reverse a Number

print("=============== Reverse a Number ===============")

number = int(input("Enter a number: "))
reverse = 0

while number > 0:
    remainder = number % 10
    reverse = (reverse * 10) + remainder
    number = number // 10

print("Reversed Number:", reverse)

# Q: Check Whether a Number is a Palindrome

print("\n=============== Palindrome Check ===============")

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

# Q: Generate the Fibonacci Sequence

print("\n============== Fibonacci Sequence ==============")

n_terms = int(input("Enter the number of terms: "))
n1 = 0
n2 = 1
print("Fibonacci Sequence:")
for i in range(n_terms):
    print(n1)
    nth = n1 + n2
    n1 = n2
    n2 = nth

# Q: Check Whether a Number is Prime

print("\n================= Prime Number =================")

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

# Q: Find All Prime Numbers Between 1 and 100

print("\n=========== Prime Numbers (1 to 100) ===========")

for num in range(1, 101):
    if num > 1:
        for i in range(2, num):
            if num % i == 0:
                break
        else:
            print(num)

print("===============================================")