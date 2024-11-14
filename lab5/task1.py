import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# 1.Загрузите данные из файла «weather1.csv» о погоде в Перми. Загрузите только следующие столбцы:
# a. Местное время в Перми
# b. T (температура воздуха в градусах Цельсия)
# c. P (атмосферное давление в мм.рт.ст.)
# d. U (относительная влажность в %)
# e. Ff (скорость ветра в м/с)
# f. N (облачность)
# g. H (высота основания облаков, м)
# h. VV (горизонтальная дальность видимости в км)

# Берем нужные колоноки
columns = [
    'Местное время в Перми', 'T', 'P', 'U', 'Ff', 'N', 'H', 'VV'
]
table = pd.read_csv('res/weather1.csv', sep = ";", usecols = columns)


# 2. Постройте точечную диаграмму (диаграмму рассеяния) по признакам
# температуры и относительной влажности.
# Создаем точечную диаграмму
# И сразу
# 3. На построенной в предыдущем пункте диаграмме выделите точки разными
# цветами в зависимости от облачности: синим – для которых облачность
# составляет 100%; красным – все остальные.
N100 = table['N'] == "100%." # условие для поиска 100% влажности
notN100 = table['N'] != "100%."

#1 график (matplotlib.pyplot)
plt.figure(figsize = (8, 6))
plt.scatter(table[N100]['T'], table[N100]['U'], color = 'blue')
plt.scatter(table[notN100]['T'], table[notN100]['U'], color = 'red')
plt.title('Диаграмма рассеяния по температуре и влажности')
plt.xlabel('Температура (в градусах)')
plt.ylabel('Влажность (в %)')
plt.show()

#2 график (seaborn)
plt.figure(figsize = (8, 6))
sns.scatterplot(
    data = table,
    x = 'T',
    y = 'U',
    hue = table['N'] == '100%.',
    palette = {True: 'blue', False: 'red'},
    legend = False,
).set(
    title='Диаграмма рассеяния по температуре и влажности',
    xlabel='Температура (в градусах)',
    ylabel='Влажность (в %)',
)
plt.show()


# 4. Постройте линейную диаграмму (график) изменения температуры в
# зависимости от местного времени.
table['Местное время в Перми'] = pd.to_datetime(table['Местное время в Перми'], dayfirst=True) # сортировка по времени

# еще один способ
# table.plot(
#     x = 'Местное время в Перми',
#     y = 'T',
#     kind = 'line',
#     figsize=(30, 6),
#     title = 'Температура в зависимости от времени',
# )
# plt.xlabel('Время')
# plt.ylabel('Температура, (в градусах)')
# plt.show()

#1 график (matplotlib.pyplot)
plt.figure(figsize = (30, 6))
plt.plot(table['Местное время в Перми'], table['T'])
plt.title('Температура в зависимости от времени')
plt.xlabel('Время')
plt.ylabel('Температура, (в градусах)')
plt.show()

#2 график (seaborn)
plt.figure(figsize = (30, 6))
sns.lineplot(
    data = table,
    x = 'Местное время в Перми',
    y = 'T',
).set(
    title = 'Температура в зависимости от времени',
    xlabel = 'Время',
    ylabel = 'Температура (в градусах)',
)
plt.show()


# 5. Посчитайте по имеющимся данным среднемесячную температуру и
# постройте столбчатую диаграмму (вертикальную) зависимости
# средней температуры от месяца. Подсказка: создайте отдельный
# столбец с номером месяца (вычислив его из столбца «Местное время»),
# а затем сгруппируйте данные по этому столбцу.
table['Месяц'] = table['Местное время в Перми'].dt.month # создание столбца с номером месяца
tempAvg = table.groupby('Месяц')['T'].mean().reset_index() # группировка данных по новому столбцу

#1 график (matplotlib.pyplot)
plt.figure(figsize = (8, 6))
plt.bar(table['Месяц'], table['T'])
plt.title('Среднемесячная температура')
plt.xlabel('Месяц')
plt.ylabel('Температура (в градусах)')
plt.show()

#2 график (seaborn)
plt.figure(figsize = (8, 6))
sns.barplot(
    data = tempAvg,
    x ='Месяц',
    y ='T'
).set(
    title='Среднемесячная температура',
    xlabel='Месяц',
    ylabel='Температура (в градусах)'
)
plt.show()


# 6. Постройте ленточную диаграмму (горизонтальную), отразив на ней
# количество имеющихся наблюдений для каждого варианта облачности.
countOfCloudness = table['N'].value_counts().reset_index()
countOfCloudness.columns = ['Облачность', 'Количество']

#1 график (matplotlib.pyplot)
plt.figure(figsize = (14, 6))
plt.barh(countOfCloudness['Облачность'], countOfCloudness['Количество'])
plt.title('Наблюдения облачности')
plt.xlabel('Количество наблюдений')
plt.show()

#2 график (seaborn)
plt.figure(figsize = (14, 6))
sns.barplot(
    data = countOfCloudness,
    x = 'Количество',
    y = 'Облачность',
    orient = 'h',
).set(
    title = 'Наблюдения облачности',
    xlabel = 'Количество наблюдений',
)
plt.show()

# 7. Постройте гистограмму частот для температуры. На гистограмме
# должно быть 10 диапазонов температуры.

#1 график (matplotlib.pyplot)
plt.figure(figsize = (8, 6))
plt.hist(table['T'], bins = 10, edgecolor = 'black')
plt.title('Частоты температур')
plt.xlabel('Температура (в градусах)')
plt.ylabel('Частота')
plt.show()

#2 график (seaborn)
plt.figure(figsize = (8, 6))
sns.histplot(table['T'], bins = 10).set(
    title = 'Частоты температур',
    xlabel = 'Температура (в градусах)',
    ylabel = 'Частота'
)
plt.show()

#8. ---

# 9. Постройте круговую диаграмму для признака «высота основания облаков».
counts = table['H'].value_counts()

#1 график (matplotlib.pyplot)
plt.figure(figsize=(8, 6))
plt.pie(
    counts,
    labels = counts.index,
    autopct = '%1.1f%%',
)
plt.title('Диаграмма для высоты основания облаков')
plt.show()

#2 график (seaborn)
plt.figure(figsize=(8, 6))
counts.plot.pie(
    autopct='%1.1f%%',
    title='Диаграмма для высоты основания облаков',
    ylabel=''
)
plt.show()