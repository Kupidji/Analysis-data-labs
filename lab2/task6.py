import re
import requests

def getMboxData():
    # Получает текст mbox файла с указанного URL
    response = requests.get('http://www.py4inf.com/code/mbox.txt')
    return response.text

def extractSenderAddresses(mboxText):
    # Извлекает адреса отправителей из текста mbox файла
    # Разделяем текст на строки
    lines = mboxText.split('\n')
    senderCounts = {}
    regex = r"^From ([^\s]+) " # До первого пробела

    for line in lines:
        # Проверяем, что строка начинается с "From "
        if not line.startswith("From "):
            continue

        # Ищем адрес отправителя
        matches = re.finditer(regex, line, re.MULTILINE)
        for match in matches:
            senderAddress = match.group(1)
            senderCounts[senderAddress] = senderCounts.get(senderAddress, 0) + 1

    return senderCounts

def getTopSender(senderCounts):
    # Находит отправителя, который написал больше всех сообщений
    # Находим отправителя с наибольшим числом сообщений
    topSender = max(senderCounts.items(), key=lambda x: x[1])
    return topSender

def printSendersInfo(senderCounts):
    # Выводит информацию о количестве сообщений у каждого отправителя
    sortedSenders = sorted(senderCounts.items(), key=lambda x: x[0])

    print(f"Адреса авторов сообщений ({len(senderCounts)}):")
    for sender, count in sortedSenders:
        print(f"- '{sender}', сообщений: {count}")

def solution():
    # Получаем текст mbox
    mboxText = getMboxData()

    # Извлекаем адреса отправителей и их количество сообщений
    senderCounts = extractSenderAddresses(mboxText)

    # Печать информации об отправителях
    printSendersInfo(senderCounts)

    # Определение отправителя с наибольшим числом сообщений
    topSender = getTopSender(senderCounts)

    print()
    print(f"Больше всех сообщений написал '{topSender[0]}': {topSender[1]} шт.")

solution()