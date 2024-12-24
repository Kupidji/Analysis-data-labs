import numpy as np
import pandas as pd
import plotly_express as px
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score


# Методы для решения лабораторной работы #

# Загрузка данных
def loadData(filepath):
    dataframe = pd.read_csv(filepath)

    return dataframe

# Проверка и обработка данных
def checkIsDataTypeNumber(dataframe):
    print("Проверка данных")
    if not dataframe.select_dtypes(exclude=["number"]).columns.empty: print("Найдены не числовые данные")
    else:
        print("Все данные - числовые")
        print(dataframe.dtypes)
        printLine()

def checkMissingDataAndFillMedian(dataframe):
    if dataframe.isnull().sum().sum() < 0:
        print("Пропущенные данные не найдены")
    else:
        print("Заполняем пропущенные значения медианой")
        dataframe.fillna(dataframe.median(), inplace=True)

    return dataframe

# Построение тепловой карты корреляции
def showPlotCorrelationHeatmap(correlation_matrix):
    px.imshow(
        correlation_matrix,
        text_auto=True,
        title='Тепловая карта для корреляционной матрицы',
        color_continuous_scale='Greens'
    ).show()

# Построение диаграммы рассеяния
def showPlotScatter(dataframe, features, target):
    for feature in features:
        plt.figure(figsize=(10, 6))
        plt.scatter(dataframe[feature], dataframe[target], color='green')
        plt.title(f'Взаимосвязь {feature} - {target}')
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.show()

# Построение боксплота
def showPlotBoxplot(dataframe, column, title):
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        dataframe[column],
        patch_artist=True,
        boxprops=dict(color="green", facecolor="lightgreen"),
        vert=False
    )
    plt.title(title)
    plt.xlabel(column)
    plt.show()

# Удаление выбросов
def removeOutliers(dataframe, column):
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    filteredDataframe = dataframe[(dataframe[column] >= q1 - 1.5 * iqr) & (dataframe[column] <= q3 + 1.5 * iqr)]

    return filteredDataframe

# Обучение и оценка модели
def trainAndEvaluateModel(model, xTrain, xTest, yTrain, yTest):
    model.fit(xTrain, yTrain)
    testPred = model.predict(xTest)
    trainPred = model.predict(xTrain)

    print("Тестовая выборка:")
    print('RMSE:', np.sqrt(mean_squared_error(yTest, testPred)))
    print('R^2:', r2_score(yTest, testPred))
    print("\nОбучающая выборка:")
    print('RMSE:', np.sqrt(mean_squared_error(yTrain, trainPred)))
    print('R^2:', r2_score(yTrain, trainPred))
    printLine()

    return model

def printLine():
    print('-+-------------------------------------------------------------------------------+-')


# 1. Загрузите данные из файла "boston.csv" о недвижимости в различных районах Бостона. Столбцы (признаки) имеют следующий смысл:
    # a. CRIM – уровень преступности
    # b. ZN – доля жилых земель, разделенных на участки площадью более 25 000 кв.футов
    # c. INDUS – доля площадей, не связанных с розничной торговлей
    # d. CHAS – наличие реки (1, если граничит с рекой; 0 в противном случае)
    # e. NOX – качество воздуха (концентрация оксидов азота)
    # f. RM – среднее количество комнат в доме
    # g. AGE – доля жилых помещений, построенных владельцами до 1940 года
    # h. DIS – взвешенные расстояния до пяти бостонских центров занятости
    # i. RAD – транспортная доступность (индекс доступности радиальных автомагистралей)
    # j. TAX – налоги (ставка налога на 10 000 долларов США)
    # k. PTRATIO – соотношение количества учеников и учителей
    # l. B – нормированное значение доли афроамериканцев среди жителей
    # m. LSTAT – процент населения с низким социальным статусом
    # n. MEDV – медианная цена недвижимости (тыс. $) – это и будет целевой признак
dataframe = loadData('res/boston.csv')

# 2. Проверьте, что у всех загруженных данных числовой тип.
# 3. Проверьте, есть ли по каким-либо признакам отсутствующие данные.
# Если отсутствующие данные есть – заполните их медианным значением.
checkIsDataTypeNumber(dataframe)
dataframe = checkMissingDataAndFillMedian(dataframe)

# 4. Посчитайте коэффициент корреляции для всех пар признаков.
# Подсказка: воспользуйтесь методом corr() для датафрейма, чтобы получить сразу всю корреляционную матрицу.
correlationMatrix = dataframe.corr()
print('Корреляционная матрица')
print(correlationMatrix)
printLine()

# 5. С помощью одной из библиотек визуализации постройте тепловую карту (heatmap) по корреляционной матрице.
showPlotCorrelationHeatmap(correlationMatrix)

# 6. Выберите от 4 до 6 признаков (на свое усмотрение), которые в наибольшей степени коррелируют с целевым признаком (ценой недвижимости).
target = 'MEDV'
features = ['LSTAT', 'RM', 'PTRATIO', 'INDUS', 'TAX']

# # 7. Для каждого из выбранных признаков в паре с целевым признаком постройте точечную диаграмму (диаграмму рассеяния).
# # 8. Визуально убедитесь, что связь между выбранным признаком и целевым
# # прослеживается. Если на основе графика считаете, что зависимости нет
# # – исключите этот признак из дальнейшего рассмотрения (но при этом как минимум 3 признака должно остаться в любом случае).
# # (признаки не были исключены)
showPlotScatter(dataframe, features, target)

# # 9. Сформируйте список факторных признаков и целевую переменную.
print("Факторные признаки - ", features)
print("Целевая переменная - ", target)
printLine()

# 10. Выполните разбиение датасета на обучающую и тестовую выборки в соотношении 8:2. При формировании обучающей и тестовой
# выборок строки из исходного датафрейма должны выбираться в случайном порядке.
# Подсказка: можно воспользоваться функцией train_test_split из библиотеки sklearn.model_selection.
x = dataframe[features]
y = dataframe[target]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)
print("Размер обучающей выборки - ", xTrain.shape)
print("Размер тестовой выборки - ", xTest.shape)
printLine()

# 11. Из набора линейных моделей библиотеки sklearn возьмите линейную регрессию, обучите ее на обучающем наборе.
# 12. Получите векторы прогнозных значений целевой переменной на обучающей и на тестовой выборках.
# 13. Посчитайте коэффициент детерминации (R2) и корень из среднеквадратичной ошибки (RMSE) на обучающей и на тестовой выборках.
linearModel = LinearRegression()
linearModel = trainAndEvaluateModel(linearModel, xTrain, xTest, yTrain, yTest)

# # 14. Постройте boxplot («ящик с усами») для целевого признака (MEDV). Определите, какие значения можно считать выбросами.
# # Указание. Если по диаграмме выбросы определить не смогли, то для выполнения дальнейших действий считайте выбросами значения MEDV=50.0.
showPlotBoxplot(dataframe, target, "Ящик с усами для целевого признака (MEDV)")

# # 15. Отфильтруйте исходные данные, удалив выбросы. Пересоздайте тестовую и
# # обучающую выборки, переобучите модель. Посчитайте показатели R2 и RMSE.
# # Как они изменились? О чем это говорит?
dataframe = removeOutliers(dataframe, target)
showPlotBoxplot(dataframe, target, "Ящик с усами для целевого признака (MEDV) после удаления выбросов")

x = dataframe[features]
y = dataframe[target]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

linearModel = trainAndEvaluateModel(linearModel, xTrain, xTest, yTrain, yTest)

ridgeModel = Ridge(alpha=0.01)
ridgeModel = trainAndEvaluateModel(ridgeModel, xTrain, xTest, yTrain, yTest)