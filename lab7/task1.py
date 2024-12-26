import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Function for initial filtering
def filterData(dataframe):
    dataframe = dataframe[dataframe['ВидПомещения'] == 'жилые помещения']
    dataframe = dataframe.drop(columns='ВидПомещения')
    dataframe = dataframe[(dataframe['СледующийСтатус'] == 'Свободна') | (dataframe['СледующийСтатус'] == 'Продана')]
    dataframe['СледующийСтатус'] = np.where(dataframe['СледующийСтатус'] == 'Свободна', 0, 1)
    dataframe = dataframe.drop(columns='УИД_Брони')
    return dataframe

# Function to check and ensure numeric data types
def validateNumericData(dataframe, numericColumns):
    for column in numericColumns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce')
    return dataframe

# Function for binary encoding
def encodeBinaryColumns(dataframe, binaryColumns):
    for column in binaryColumns:
        uniqueValues = dataframe[column].dropna().unique().tolist()
        uniqueValues.sort()
        dataframe[column] = dataframe[column].map({uniqueValues[0]: 1, uniqueValues[1]: 0})
    return dataframe

# Function for one-hot encoding
def encodeCategoricalColumns(dataframe, categoricalColumns):
    return pd.get_dummies(dataframe, columns=categoricalColumns)

def encodeTypeFeature(dataframe):
    dataframe['Тип'] = dataframe['Тип'].map(lambda x: str(x)[:-1].replace(',', '.'))
    dataframe['Тип'] = dataframe['Тип'].replace('', np.nan)
    dataframe['Тип'] = pd.to_numeric(dataframe['Тип'], errors="coerce")
    return dataframe


def handleMissingData(dataframe):
    print("Проверим количество отсутствующих данных")
    print(dataframe.isnull().sum())
    printLine()
    # Заметим, что в признаках: Тип, ПродаваемаяПлощадь, СтоимостьНаДатуБрони, ВариантОплаты, ВариантОплатыДоп, СкидкаНаКвартиру, ФактическаяСтоимостьПомещения
    # отсутствуют данные, заполним их
    # a
    dataframe['СкидкаНаКвартиру'] = dataframe['СкидкаНаКвартиру'].fillna(0) # почти всегда скидка 0

    # b
    dataframe['Тип'] = dataframe['Тип'].fillna(dataframe['Тип'].median())
    dataframe['ПродаваемаяПлощадь'] = dataframe['ПродаваемаяПлощадь'].fillna(dataframe['ПродаваемаяПлощадь'].median())

    # c
    dataframe = dataframe.drop(columns = ['ВариантОплатыДоп']) # я решил полностью убрать столбец из рассмотрения
    print("Проверим количество отсутствующих данных после заполнения")
    print(dataframe.isnull().sum())
    printLine()

    # d
    dataframe = dataframe.dropna() # заметим, что оставшихся незаполненных данных не так уж и много, просто удалим их
    return dataframe

# Function for feature engineering
def addFeatures(dataframe):
    dataframe['ЦенаЗаКвадратныйМетр'] = dataframe['ФактическаяСтоимостьПомещения'] / dataframe['ПродаваемаяПлощадь']
    dataframe['СкидкаВПроцентах'] = (dataframe['СкидкаНаКвартиру'] / dataframe['ФактическаяСтоимостьПомещения']) * 100
    print('Проверим добавленные признаки')
    # Настроить параметры отображения
    pd.set_option('display.max_columns', None)  # Показать все столбцы
    pd.set_option('display.max_rows', None)  # Показать все строки
    pd.set_option('display.width', 1000)
    print(dataframe.tail(5))
    pd.reset_option('all', silent = True)
    printLine()
    return dataframe

# Function for normalization
def normalizeData(dataframe, discountColumn, numericColumns):
    dataframe[discountColumn] = MinMaxScaler(feature_range=(-0.5, 0.5)).fit_transform(dataframe[[discountColumn]])
    dataframe[numericColumns] = MinMaxScaler(feature_range=(0, 1)).fit_transform(dataframe[numericColumns]) # Выполню нормализацию по умолчанию
    return dataframe

# Function to train and evaluate models
def trainAndEvaluateModels(xTrain, xTest, yTrain, yTest):
    knnModel = KNeighborsClassifier().fit(xTrain, yTrain)
    treeModel = DecisionTreeClassifier().fit(xTrain, yTrain)

    predictions = {
        "knn": (knnModel.predict(xTrain), knnModel.predict(xTest)),
        "tree": (treeModel.predict(xTrain), treeModel.predict(xTest)),
    }

    metrics = {}
    for model, (trainPred, testPred) in predictions.items():
        metrics[model] = {
            "Обучение": {
                "precision": precision_score(yTrain, trainPred),
                "recall": recall_score(yTrain, trainPred),
                "f": f1_score(yTrain, trainPred),
            },
            "Тест": {
                "precision": precision_score(yTest, testPred),
                "recall": recall_score(yTest, testPred),
                "f": f1_score(yTest, testPred),
            },
        }
    return metrics

def printLine():
    print('-+-------------------------------------------------------------------------------+-')

# 1. Загрузите данные из файла "база.csv". Столбцы (признаки) имеют следующий смысл:
    # a. УИД_Брони – уникальный идентификатор брони (для классификации его использовать не надо – это просто id)
    # b. ДатаБрони – дата установления брони
    # c. ВремяБрони – время установления брони (GMT+4)
    # d. ИсточникБрони – способ оформления брони: МП — мобильное приложение, ручная — офис продаж (бинарный признак)
    # e. ВременнаяБронь – подтвержденная заявка на бронь(«Нет») или заявка на бронь не подтверждена отделом продаж («Да») (бинарный признак)
    # f. СледующийСтатус – если пусто или «В резерве» — статус еще не определен, «Продана» — договор оформлен, «Свободна» — бронь снята (отменена). Это и есть целевой признак, который нужно научиться предсказывать
    # g. Город – город проекта
    # h. ВидПомещения – один из 4 видов: жилые помещения, нежилые помещения, кладовые, паркинг. Нас будут интересовать бронирования только со значением этого признака «жилые помещения».
    # i. Тип – количество комнат для жилых помещений (для остальных видов неактуально)
    # j. ПродаваемаяПлощадь – общая продаваемая площадь помещения
    # k. Этаж – этаж расположения помещения
    # l. СтоимостьНаДатуБрони – актуальная стоимость по прайсу на дату установления брони
    # m. ТипСтоимости – «Стоимость при 100% оплате» или «Стоимость в рассрочку», остальные варианты можно считать отсутствием значения (бинарный признак)
    # n. ВариантОплаты – «Единовременная оплата» или «Оплата в рассрочку» (бинарный признак)
    # o. ВариантОплатыДоп – уточненный вариант оплаты: Ипотека/Вторичное жилье, если "пусто" — использовать ВариантОплаты
    # p. СкидкаНаКвартиру – размер предоставленной скидки при её наличии, отрицательное значение — это наценка на помещение (как правило, при
    # выводе из закрытого ассортимента)
    # q. ФактическаяСтоимостьПомещения – стоимость заключения сделки (стоимость на дату брони за вычетом скидки)
    # r. СделкаАН – участие в сделке агентства недвижимости (да/нет) (бинарный признак)
    # s. ИнвестиционныйПродукт – да/нет — признак продажи помещения по инвестиционному договору (бинарный признак)
    # t. Привилегия – да/нет — признак продажи помещения по инвестиционному договору типа Привилегия (бинарный признак)
    # u. Статус лида (из CRM) – Статус клиента: S - успешный, P - в работе, F - забракованный
dataframe = pd.read_csv('res/data7lab.csv', encoding='windows-1251', sep=';')

# 2. Предварительная фильтрация.
    # a. Поскольку нас интересуют только сделки с жилой недвижимостью, отфильтруйте данные, оставив только те,
    # для которых «ВидПомещения» = «жилые помещения». В дальнейшем этот столбец использоваться не будет,
    # его можно удалить (или удалите его из датасета вообще, или просто нигде далее не рассматривайте).
    # b. Также для нас бесполезны данные, по которым статус не определен. Отфильтруйте данные по признаку «СледующийСтатус».
    # В оставшихся строчках замените значение «Продана» на 1, «Свободна» – на 0.
    # c. Не забывайте, что столбец «УИД_Брони» для нас также не представляет интереса – удалите его из
    # датасета вообще, или просто нигде далее не рассматривайте.
dataframe = filterData(dataframe)
print('Проверим предварительную фильтрацию')
pd.set_option('display.max_columns', None)  # Показать все столбцы
pd.set_option('display.max_rows', None)  # Показать все строки
pd.set_option('display.width', 1000)
print(dataframe.head(5))
pd.reset_option('all', silent = True)
printLine()

# 3. Проверьте тип данных и преобразуйте все данные к числовому типу.
    # a. Для тех полей, которые по смыслу являются числовыми (например, «ПродаваемаяПлощадь») – просто проверьте правильность типа.
    # b. Для бинарных признаков (например, «ИсточникБрони») выполните кодирование (один вариант закодируйте 0, другой 1).
    # c. Для категориальных не бинарных признаков (например, «Город») выполните one-hot кодирование.
    # d. Обратите внимание на поле «Тип». По смыслу оно числовое (количество комнат), но напрямую сконвертировать его
    # в числовой тип мешает буковка «к» в конце. Напишите вручную преобразование, которое удаляет букву «к» в конце и конвертирует то,
    # что осталось, в число. Если это невозможно (среди данных вам встретится еще вариант, когда в этом поле
    # записано просто «с») – просто пока оставьте поле пустым (NaN).
print("Проверим типы данных:")
print(dataframe.dtypes)
printLine()

# a
numericColumns = ['ПродаваемаяПлощадь', 'Этаж', 'СтоимостьНаДатуБрони', 'СкидкаНаКвартиру', 'ФактическаяСтоимостьПомещения']
dataframe = validateNumericData(dataframe, numericColumns)

# b
binaryColumns = ['ИсточникБрони', 'ВременнаяБронь', 'ТипСтоимости', 'ВариантОплаты', 'СделкаАН', 'ИнвестиционныйПродукт', 'Привилегия']
dataframe = encodeBinaryColumns(dataframe, binaryColumns)

# c
categoricalColumns = ['Город', 'Статус лида (из CRM)']
dataframe = encodeCategoricalColumns(dataframe, categoricalColumns)
# d
dataframe = encodeTypeFeature(dataframe)

print("Проверим типы данных после преобразования:")
print(dataframe.dtypes)
# Так как ВариантОплатыДоп будет удален на следующем шаге, я решил его не трогать
printLine()

# 4. Проверьте, есть ли по каким-либо признакам отсутствующие данные.
    # a. Отсутствующие данные в поле «СкидкаНаКвартиру» замените на 0 (это значение по умолчанию – если поле не заполнено, то скидки, по всей видимости, нет).
    # b. Отсутствующие данные в полях «Тип» и «ПродаваемаяПлощадь» замените на медианное значение, вычисленное по всему
    # набору данных (признаки кажутся достаточно важными, поэтому удалять эти столбцы не хочется; пустых значений довольно много,
    # поэтому удалять строки тоже не очень хорошо; какого-то значения «по умолчанию» для этих полей нет; поэтому заменить эти
    # значения на медиану представляется наилучшим решением).
    # c. Что делать с полем «ВариантОплатыДоп» решите самостоятельно (можно, как указано в описании, вместо пустых значений
    # использовать значение из поля «ВариантОплаты», но в таком случае обратите внимание, что признак становится не бинарным;
    # допустимо также совсем убрать этот столбец из рассмотрения).
    # d. По всем остальным полям примите решение самостоятельно. Если отсутствующих данных не много, то удалите соответствующие строки.
dataframe = handleMissingData(dataframe)

# 5. Дополнение данных.
    # a. Добавьте новый признак «Цена за квадратный метр». Он должен вычисляться на основе значений признаков «ФактическаяСтоимостьПомещения» и «ПродаваемаяПлощадь».
    # b. Добавьте новый признак «Скидка в процентах», на основе значений «ФактическаяСтоимостьПомещения» и «СкидкаНаКвартиру».
    # Комментарий. Многие алгоритмы классификации, по сути, учитывают только линейную зависимость. Поэтому, несмотря на то что
    # добавляемые признаки полностью определяются значениями других признаков, их добавление имеет смысл – их зависимость от других признаков не линейная.
dataframe = addFeatures(dataframe)

# 6. Выполните нормализацию. Можете самостоятельно выбрать способ нормализации. «По умолчанию» предлагается выполнить
# минимаксную нормализацию и привести все значения к диапазону [0;1], кроме признака «СкидкаНаКвартиру» - его логичнее приводить к диапазону [-0,5; 0,5].
numericColumns.extend(['Тип', 'ЦенаЗаКвадратныйМетр', 'СкидкаВПроцентах'])
numericColumns.remove('СкидкаНаКвартиру')
print("Текущие признаки для нормализации:", numericColumns)
printLine()
dataframe = normalizeData(dataframe, 'СкидкаНаКвартиру', numericColumns)

# # 7. Проверьте датасет на сбалансированность (количество строк со значением целевого признака 0 и со значением 1). Является ли датасет сбалансированным?
print(dataframe["СледующийСтатус"].value_counts())
printLine()
# Датасет не сбалансирован, но по заданию не сказано делать какие-либо манипуляции, продолжим решение

# 8. Сформируйте список факторных признаков и целевой признак.
target = 'СледующийСтатус'
features = list(dataframe.columns)
features.remove(target)
features.remove('ДатаБрони') # Исключаем дату и время так как они нам не понадобятся
features.remove('ВремяБрони')

# 9. Выполните разбиение датасета на обучающую и тестовую выборки. При формировании обучающей и тестовой выборок строки из
# исходного датафрейма должны выбираться в случайном порядке.
x = dataframe[features]
y = dataframe[target]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)
print("Размер обучающей выборки:", xTrain.shape[0])
print("Размер тестовой выборки:", xTest.shape[0])
printLine()

# 10. Из библиотеки sklearn.neighbors возьмите алгоритм классификации KNN (KNeighborsClassifier). Постройте (обучите) модель.
# Для параметров используйте значения по умолчанию.
# 11. Из библиотеки sklearn.tree возьмите алгоритм классификации на основе деревьев решений (DecisionTreeClassifier).
# Постройте (обучите) модель. Для параметров используйте значения по умолчанию.
# 12. Получите векторы прогнозных значений целевой переменной на обучающей и на тестовой выборках для каждой из моделей.
# 13. Посчитайте показатели качества: «F-мера», точность (Precision) и полнота (Recall) на обучающей и на тестовой выборках для каждой из моделей.

# Снимем метрики для двух моделей
metrics = trainAndEvaluateModels(xTrain, xTest, yTrain, yTest)
for model, results in metrics.items():
    print(f"{model.upper()} Результаты:")
    for dataset, scores in results.items():
        print(f"  {dataset.capitalize()}: Precision (точность) = {scores['precision']:.4f}, Recall (полнота) = {scores['recall']:.4f}, F-мера = {scores['f']:.4f}")
    printLine()

# 14. Сделайте вывод о том, насколько хорошо удалось решить задачу прогнозирования. Какая модель оказалась лучше?
# Дайте интерпретацию полученных значений Precision и Recall.

# Сняв метрики я сделал следующие выводы:
# 1) Tree подходит к этой задаче намного лучше, чем knn, так как и на обучении и на тесте показывает более стабильные и хорошие результаты
# 2) Судя по результатам обучения tree переобучена, точность равна 1, также все остальные метрики очень близки к 1

# Интерпретация Precision и Recall.
# Precision - измеряет насколько верно модель спрогнозировала, что значение является истинным (0 до 1, можно сказать, что это вероятность)
# Recall - измеряет насколько точно модель нашла все фактически истинные значения в наборе данных