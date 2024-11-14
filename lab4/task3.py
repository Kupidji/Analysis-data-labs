import pandas as pd

# Берем нужные колоноки
columns = [
    'State', 'Area code', 'International plan', 'Number vmail messages',
    'Total day minutes', 'Total day calls', 'Total eve minutes', 'Total eve calls',
    'Total night minutes', 'Total night calls', 'Customer service calls', 'Churn',
]
table = pd.read_csv('res/telecom_churn.csv', usecols = columns)

# 1. Выведите общую информацию о датафрейме с помощью методов info или describe.
# Есть ли отсутствующие данные?
print(table.describe())
missing = table.isnull().sum().sum()
print(f"Отсутствующие данные - {missing}")
print("\n")

# 2. С помощью метода value_counts определите, сколько клиентов активны,
# а сколько потеряно. Сколько процентов клиентов в имеющихся данных активны,
# а сколько потеряны?
total = table.shape[0]
active = table['Churn'].value_counts()[False]
lost = table['Churn'].value_counts()[True]
print(f"Активных клиентов: {active} = {active / total * 100:.2f}%")
print(f"Потеряно клиентов: {lost} = {lost / total * 100:.2f}%")

# 3. Добавьте дополнительный столбец в датафрейм – средняя продолжительность
# одного звонка (вычислить как суммарная продолжительность всех звонков,
# деленная на суммарное количество всех звонков). Отсортируйте данные по
# этому значению по убыванию и выведите 10 первых записей.
totalMinutes = (table['Total day minutes'] + table['Total eve minutes'] + table['Total night minutes'])
totalCalls = (table['Total day calls'] + table['Total eve calls'] + table['Total night calls'])
table['Average duration'] = totalMinutes / totalCalls
sortByAvgDur = table.sort_values(by = 'Average duration', ascending = False)
print("Первые 10 записей по средней продолжительности одного звонка:")
print(sortByAvgDur.head(10))
print("\n")

# 4. Сгруппируйте данные по значению поля «Churn» и вычислите среднюю
# продолжительность одного звонка в каждой категории. Есть ли существенная
# разница в средней продолжительности одного звонка между активными и
# потерянными клиентами?
groupByChurnAD = table.groupby('Churn')['Average duration'].mean()
print("Средняя продолжительность одного звонка:")
print(f"- Активным клиентам = {groupByChurnAD[False]:.2f}")
print(f"- Потерянным клиентам = {groupByChurnAD[True]:.2f}")

# 5. Сгруппируйте данные по значению поля «Churn» и вычислите среднее количество
# звонков в службу поддержки в каждой категории. Есть ли существенная
# разница между активными и потерянными клиентами?
groupByChurnAvgCSC = table.groupby('Churn')['Customer service calls'].mean()
print("Среднее количество звонков в службу поддержки:")
print(f"- От активных клиентов = {groupByChurnAvgCSC[False]:.2f}")
print(f"- От потерянных клиентов = {groupByChurnAvgCSC[True]:.2f}")
print("\n")

# 6. Исследуйте подробнее связь между параметрами «Churn» и «Customer service
# calls», построив таблицу сопряженности (факторную таблицу) по этим
# признакам. Подсказка: используйте функцию crosstab. При каком
# количестве звонков в службу поддержки процент оттока становится
# существенно выше, чем в целом по датафрейму? (В качестве уточнения
# фразы «существенно выше» можете использовать «более 40%».)
# таблица сопряженности
crosstab = pd.crosstab(table['Customer service calls'], table['Churn'])
print("Таблица сопряженности:")
print(crosstab)
print("\n")

churnPercent = crosstab.div(crosstab.sum(axis = 1), axis = 0) * 100
print("Процент оттока по количеству звонков в службу поддержки:")
print(churnPercent[True])
print("\n")

highChurnCalls = churnPercent[churnPercent[True] > 40]
print("Количество звонков в службу поддержки, при котором процент оттока выше:")
print(highChurnCalls[True])
print("\n")

# 7. Аналогично предыдущему пункту исследуйте связь между параметрами
# «Churn» и «International plan». Можно ли утверждать, что процент
# оттока среди клиентов, использующих международный роуминг, существенно
# выше или ниже, чем среди клиентов, не использующих его?
# таблица сопряженности
crosstab = pd.crosstab(table['International plan'], table['Churn'])
print("Таблица сопряженности:")
print(crosstab)
print("\n")

churnPercent = crosstab.div(crosstab.sum(axis = 1), axis = 0) * 100
print("Процент оттока для клиентов с международным роумингом и без него:")
print(churnPercent[True])