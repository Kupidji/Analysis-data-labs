def solution(s):
    result = []
    k = 1
    for i in range(0, len(s) - 1):
        if (s[i] == s[i + 1]):
            k = k + 1
            continue

        if (k > 1):
            newStr = str(k) + s[i]
            result += newStr
            k = 1
        else:
            result += s[i]
            k = 1

    if (k > 1):
        if (s[len(s)-2] == s[len(s)-1]):
            newStr = str(k) + s[len(s)-1]
            result += newStr
    else:
        result += s[len(s)-1]

    return result

s = str(input())
print("".join(solution(s)))