from collections import defaultdict

def calculateBalances(participants, expenses):
    # Вычисляет баланс каждого участника по сравнению со средней тратой
    totalExpenses = sum(expenses.values())
    averageExpense = totalExpenses / len(participants)

    # Вычисляется сумма всех расходов участника, и минусуются средние траты
    balances = {participant: round(expenses[participant] - averageExpense, 2) for participant in participants}
    return balances

def findDebtsAndCredits(balances):
    # Определяет, кто кому должен и кто сколько должен получить
    debts = defaultdict(float)
    credits = defaultdict(float)

    for participant, balance in balances.items():
        if balance < 0:
            debts[participant] = -balance  # Долг участника
        elif balance > 0:
            credits[participant] = balance  # Кредит участника

    return debts, credits

def settleDebts(debts, credits):
    # Рассчитывает минимальное количество переводов для погашения долгов

    # Список для вывода результата (должник, кому должен, сумма)
    transactions = []

    # Погашение долгов с равными значениями
    # Проходится по всем должникам и кредиторам, если их долг-кредит равны, то их
    # значения обнуляются и они добавляются в список transactions
    for debtor, debtAmount in list(debts.items()):
        for creditor, creditAmount in list(credits.items()):
            if debtAmount == creditAmount:
                transactions.append((debtor, creditor, debtAmount))
                debts[debtor] = 0
                credits[creditor] = 0
                break

    # Погашение долгов с неравными значениями
    # Рассчитывается минимальная сумма для перевода
    # (transferAmount = min(debts[debtor], credits[creditor])).
    while debts and credits:
        debtor = next((name for name, amount in debts.items() if amount > 0), None)
        creditor = next((name for name, amount in credits.items() if amount > 0), None)

        if debtor is None or creditor is None:
            break

        transferAmount = min(debts[debtor], credits[creditor])
        transactions.append((debtor, creditor, transferAmount))

        debts[debtor] -= transferAmount
        credits[creditor] -= transferAmount

        if debts[debtor] == 0:
            del debts[debtor]
        if credits[creditor] == 0:
            del credits[creditor]

    return transactions


# Чтение входных данных
participants = input("Введите имена участников через пробел: ").strip().split()
numPurchases = int(input("Введите количество покупок: ").strip())

# Чтение расходов
expenses = defaultdict(float)
for _ in range(numPurchases):
    name, amount = input("Введите имя и сумму покупки через пробел: ").strip().split()
    amount = float(amount)
    expenses[name] += amount

# Вычисление балансов участников
balances = calculateBalances(participants, expenses)

# Определение долгов и кредитов
debts, credits = findDebtsAndCredits(balances)

# Рассчет минимального количества переводов для погашения долгов
transactions = settleDebts(debts, credits)

print(len(transactions))
for debtor, creditor, amount in transactions:
    print(f"{debtor} должен {creditor} {amount:.2f} рублей")