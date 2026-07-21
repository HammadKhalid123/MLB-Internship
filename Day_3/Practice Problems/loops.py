# Q: Print Numbers from 1 to 100

print("============== Numbers from 1 to 100 ==============")

for i in range(1, 101):
    print(i, end=" ")

print("\n============= Even Numbers (1 to 100) =============")

for i in range(1, 101):
    if i % 2 == 0:
        print(i, end=" ")

# Q: Calculate the Sum of Numbers from 1 to N

print("\n============== Sum of Numbers =====================")

number = int(input("Enter a number: "))
sum = 0
for i in range(1, number + 1):
    sum = sum + i

print("The sum of numbers from 1 to", number, "is", sum)

# Q: Print the Multiplication Table of a Given Number

print("\n============= Multiplication Table ================")

number = int(input("Enter a number: "))
for i in range(1, 11):
    print(number, "x", i, "=", number * i)

# Q: Count the Number of Digits in a Number

print("\n================ Count Digits =====================")

number = int(input("Enter a number: "))
count = 0
temp = number

if temp == 0:
    count = 1
else:
    while temp > 0:
        temp = temp // 10
        count += 1
print("Total Digits:", count)

print("===================================================")