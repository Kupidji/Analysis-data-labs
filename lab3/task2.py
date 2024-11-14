import numpy as np

# Задание 2

# 1. Сгенерировать два одномерных массива m1 и m2 размером 10
m1Array = np.random.randint(1, 11, 10)
m2Array = np.random.randint(1, 11, 10)
print("\nМассив m1:\n", m1Array)
print("Массив m2:\n", m2Array)

# 2. Сформировать массив m3, содержащий элементы, которые есть только в m1 или только в m2
m3Array = np.setxor1d(m1Array, m2Array)
print("\nМассив m3 (элементы, которые есть либо только в m1, либо только в m2):\n", m3Array)

# 3. Заменить в m1 все значения, кратные 3 или 2 на 1
m1ModArray = np.where((m1Array % 3 == 0) | (m1Array % 2 == 0), 1, m1Array)
print("\nМассив m1 после замены кратных 2 или 3 на 1:\n", m1ModArray)

# 4. Слить m1 и m2 в один и преобразовать в матрицу размером 4x5
mergedArray = np.hstack((m1Array, m2Array))
mergedMatrix = mergedArray.reshape(4, 5)
print("\nМатрица (4x5) после объединения m1 и m2:\n", mergedMatrix)

# 5. Удалить 1 и 4 столбцы
matrixWithoutColumns = np.delete(mergedMatrix, [0, 3], axis=1)
print("\nМатрица после удаления 1 и 4 столбцов:\n", matrixWithoutColumns)

# 6. Транспонировать полученную матрицу
transpMatrix = matrixWithoutColumns.T
print("\nТранспонированная матрица:\n", transpMatrix)
