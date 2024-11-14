import re

def checkPassword(password):
    if len(password) < 12:
        return False

    return (
        re.search(r'[A-Z]', password) and
        re.search(r'[a-z]', password) and
        re.search(r'[0-9]', password) and
        re.search(r'[!@#$%&*+]', password)
    )


n = int(input())
for _ in range(n):
    password = input().strip()
    if checkPassword(password):
        print("Valid")
    else:
        print("Invalid")