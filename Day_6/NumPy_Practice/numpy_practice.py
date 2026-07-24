import numpy as np


arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]])

print("===== 1D Array =====")
print(arr_1d)

print("\n===== 2D Array =====")
print(arr_2d)

print("\n===== Arithmetic Operations =====")
print("Addition (+5):", arr_1d + 5)
print("Subtraction (-1):", arr_1d - 1)
print("Multiplication (*2):")
print(arr_2d * 2)
print("Division (/2):", arr_1d / 2)


print("\n===== Array Statistics =====")
print("1D Array")
print(f"Maximum: {np.max(arr_1d)}")
print(f"Minimum: {np.min(arr_1d)}")
print(f"Mean: {np.mean(arr_1d):.2f}")
print(f"Sum: {np.sum(arr_1d)}")


print("\n2D Array")
print(f"Maximum: {np.max(arr_2d)}")
print(f"Minimum: {np.min(arr_2d)}")
print(f"Mean: {np.mean(arr_2d):.2f}")
print(f"Sum: {np.sum(arr_2d)}")


reshaped_1d = arr_1d.reshape(5, 1)
reshaped_2d = arr_2d.reshape(3, 2)

print("\n===== Reshaped Arrays =====")
print("1D Array (5 x 1):")
print(reshaped_1d)

print("\n2D Array (3 x 2):")
print(reshaped_2d)


print("\n===== Indexing and Slicing =====")
print("1D Indexed Element (Index 2):", arr_1d[2])
print("2D Indexed Element (Row 1, Column 2):", arr_2d[1, 2])

print("\n1D Sliced Array (Index 1 to 3):")
print(arr_1d[1:4])
print("\n2D Sliced Array:")
print(arr_2d[0:2, 1:3])