
# Question 1: Find Unique Values from a List


numbers = [1, 2, 2, 3, 4, 4, 5, 6, 6, 7]

# Logic Approach
unique_numbers = []

for num in numbers:
    if num not in unique_numbers:
        unique_numbers.append(num)

print("=" * 50)
print("Question 1: Find Unique Values from a List")
print("=" * 50)
print("Logic Approach")
print("Unique Values:", unique_numbers)

# Built-in Method
unique_builtin = set(numbers)

print("\nBuilt-in Method")
print("Unique Values:", unique_builtin)

# Question 2: Perform Union Operation

set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

# Logic Approach
union_set = set1.copy()

for item in set2:
    if item not in union_set:
        union_set.add(item)

print("\n" + "=" * 50)
print("Question 2: Union of Two Sets")
print("=" * 50)
print("Logic Approach")
print("Union:", union_set)

# Built-in Method
print("\nBuilt-in Method")
print("Union:", set1.union(set2))

# Question 3: Perform Intersection Operation

# Logic Approach
intersection_set = set()

for item in set1:
    if item in set2:
        intersection_set.add(item)

print("\n" + "=" * 50)
print("Question 3: Intersection of Two Sets")
print("=" * 50)
print("Logic Approach")
print("Intersection:", intersection_set)

# Built-in Method
print("\nBuilt-in Method")
print("Intersection:", set1.intersection(set2))