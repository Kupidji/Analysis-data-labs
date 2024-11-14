import numpy as np

# 1. Сгенерировать случайным образом матрицу размерностью 8x8
matrix = np.random.randint(-10, 11, (8, 8))
print("Матрица 8x8:\n", matrix)

# 2. Вывести центральную часть матрицы размером 4x4
centralPart = matrix[2:6, 2:6]
print("\nЦентральная часть матрицы 4x4:\n", centralPart)

# 3. Удалить из исходной матрицы все строки, содержащие минимальный элемент
minElement = matrix.min()
rowsWithMin = np.any(matrix == minElement, axis=1)
matrixWithoutMinRows = matrix[~rowsWithMin]
print("\nМатрица без строк с минимальным элементом:\n", matrixWithoutMinRows)

# 4. Вставить строку из минимального элемента перед первой строкой
minRow = np.full((1, 8), minElement)
matrixWithMinRow = np.vstack((minRow, matrix))
print("\nМатрица с добавленной строкой минимальных элементов:\n", matrixWithMinRow)

# 5. Вычислить сумму и среднее арифметическое всех элементов матрицы
sumMatrix = matrix.sum()
meanMatrix = matrix.mean()
print("\nСумма элементов матрицы:", sumMatrix)
print("Среднее арифметическое элементов матрицы:", meanMatrix)
