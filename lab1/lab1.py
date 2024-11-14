# def solution(N):
#     result = 0
#     for i in range(1, N+1):
#         result += 2*i / (i+2)
#     return round(result, 3)
#
# N = int(input())
# result = solution(N)
# print(result)

# def solution(x, y, n):
#     step = x - y
#     result = 1
#     currentValue = x
#     while (n >= currentValue):
#         currentValue += step
#         result += 1
#
#     return result
#
# X = int(input())
# Y = int(input())
# N = int(input())
# print(solution(X, Y, N))

# X = int(input())
# Y = int(input())
# N = int(input())
#
# count = 0
# pictures = 0
#
# while pictures <= N:
#     count += 1
#     pictures += X
#     if count > 1:
#         pictures -= Y
#
# print(count)


# def solution(X, Y):
#     totalTeeth = 2 * X
#     totalTime = totalTeeth * Y
#     hours = totalTime // 60
#     minutes = totalTime % 60
#     return (8 + hours, minutes)
#
# X = int(input())
# Y = int(input())
#
# result = solution(X, Y)
# print(f"{result[0]}")
# print(f"{result[1]}")

# def solution(X, Y):
#     totalPages = 0
#     day = 0
#
#     while totalPages < Y:
#         day += 1
#         pagesToday = (X + 2 * (day - 1)) * day
#         totalPages += pagesToday
#
#     return day
#
# X = int(input())
# Y = int(input())
# print(solution(X, Y))

# X = int(input())
# Y = int(input())
# N = int(input())
#
# pictures = N - X
# step = X - Y
#
# if pictures > 0:
#     days = pictures / step + 1
# else:
#     days = 1
#
# if pictures >= 0:
#     days += 1
#
# print(int(days))

# def solution(size, sum):
#     if sum < 1 or sum > 9 * size:
#         return 'NO'
#
#     result = [0] * size
#
#     for i in range(size - 1, -1, -1):
#         if i == 0:
#             result[i] = sum
#         elif sum > 1:
#             if sum > 9:
#                 result[i] = 9
#                 sum -= 9
#             else:
#                 result[i] = sum - 1
#                 sum = 1
#
#     return ''.join(map(str, result))
#
# N = int(input())
# S = int(input())
# print(solution(N, S))


def solution(N):
    while (len(set(str(N))) != len(str(N))):
        N -= 1
    return N

N = int(input())
print(solution(N))
