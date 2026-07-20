# Topic: Lists
#
print("""Note:
 Each question has been solved using:
 1. Logic Approach
 2. Python Built-in Method (where applicable)"""
)

# ==========================================
# Question 1: Find the Largest Number in a List
# ==========================================

numbers = [12, 14, 16, 10, 3, 29]

# Logic Approach
max_number = numbers[0]

for num in numbers:
    if num > max_number:
        max_number = num

print("=" * 50)
print("Question 1: Find the Largest Number")
print("=" * 50)
print("Logic Approach")
print("Largest Number:", max_number)

# Built-in Method
print("\nBuilt-in Method")
print("Largest Number:", max(numbers))


# ==========================================
# Question 2: Find the Second Largest Number
# ==========================================

if len(numbers) < 2:
    print("List should contain at least two elements.")
else:

    # Logic Approach
    second_largest = numbers[0]

    for num in numbers:
        if num > second_largest and num < max_number:
            second_largest = num

    print("\n" + "=" * 50)
    print("Question 2: Find the Second Largest Number")
    print("=" * 50)
    print("Logic Approach")
    print("Second Largest Number:", second_largest)

    # Built-in Method
    unique_numbers = list(set(numbers))
    unique_numbers.sort()

    print("\nBuilt-in Method")
    print("Second Largest Number:", unique_numbers[-2])


# ==========================================
# Question 3: Remove Duplicate Values
# ==========================================

numbers = [1, 2, 2, 3, 4, 4, 5, 6, 6, 7]

# Logic Approach
unique_numbers = []

for num in numbers:
    if num not in unique_numbers:
        unique_numbers.append(num)

print("\n" + "=" * 50)
print("Question 3: Remove Duplicate Values")
print("=" * 50)
print("Logic Approach")
print("Original List:", numbers)
print("Without Duplicates:", unique_numbers)

# Built-in Method
print("\nBuilt-in Method")
print("Without Duplicates:", list(set(numbers)))


# ==========================================
# Question 4: Reverse a List
# ==========================================

# Logic Approach

reverse_list = []

for num in numbers[::-1]:
    reverse_list.append(num)

print("\n" + "=" * 50)
print("Question 4: Reverse a List")
print("=" * 50)
print("Logic Approach")
print("Reversed List:", reverse_list)

# Built-in Method

reverse_builtin = numbers.copy()
reverse_builtin.reverse()

print("\nBuilt-in Method")
print("Reversed List:", reverse_builtin)


# ==========================================
# Question 5: Find Common Elements Between Two Lists
# ==========================================

list1 = [1, 2, 3, 4, 5]
list2 = [3, 4, 5, 6, 7]

# Logic Approach

common = []

for item in list1:
    if item in list2:
        common.append(item)

print("\n" + "=" * 50)
print("Question 5: Find Common Elements Between Two Lists")
print("=" * 50)
print("Logic Approach")
print("Common Elements:", common)

# Built-in Method

common_builtin = list(set(list1) & set(list2))

print("\nBuilt-in Method")
print("Common Elements:", common_builtin)