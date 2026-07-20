

numbers = (1, 2, 3, 2, 4, 2, 5)

# Question 1: Count Occurrences of an Element

num = int(input("Enter a number to count its occurrences: "))

# Logic Approach
count = 0

for item in numbers:
    if item == num:
        count += 1

print("=" * 50)
print("Question 1: Count Occurrences of an Element")
print("=" * 50)
print("Logic Approach")
print(f"The number {num} occurs {count} times.")

# Built-in Method
count_builtin = numbers.count(num)

print("\nBuilt-in Method")
print(f"The number {num} occurs {count_builtin} times.")

# Question 2: Convert Tuple into List

print("\n" + "=" * 50)
print("Question 2: Convert Tuple into List")
print("=" * 50)

list_from_tuple = list(numbers)

print("Tuple:", numbers)
print("Converted List:", list_from_tuple)

# Question 3: Convert List into Tuple

sample_list = [10, 20, 30, 40, 50]

print("\n" + "=" * 50)
print("Question 3: Convert List into Tuple")
print("=" * 50)

tuple_from_list = tuple(sample_list)

print("List:", sample_list)
print("Converted Tuple:", tuple_from_list)