def getMid(lst):
    size = len(lst)
    mid = size // 2
    if size % 2 == 1:
        return lst[mid]
    else:
        return (lst[mid - 1] + lst[mid]) / 2

N = int(input())
attendanceList = [int(input()) for _ in range(N)]

attendanceList.sort()

Q2 = getMid(attendanceList)

if N % 2 == 0:
    lowerPart = attendanceList[:N//2]
    upperPart = attendanceList[N//2:]
else:
    lowerPart = attendanceList[:N//2]
    upperPart = attendanceList[N//2 + 1:]

Q1 = getMid(lowerPart)
Q3 = getMid(upperPart)

IQR = Q3 - Q1

lowerBound = Q1 - 1.5 * IQR
upperBound = Q3 + 1.5 * IQR

countOfBadDigits = 0
for digit in attendanceList:
    if digit < lowerBound or digit > upperBound:
        countOfBadDigits += 1

print(countOfBadDigits)
