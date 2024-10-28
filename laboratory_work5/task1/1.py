from random import randint
from math import exp
from dsmltf import scale, mult_predict, dot, f1_score, gauss_slae

# Генерация данных о готовке
def make_cooking_data() -> list:
    """
    Генерирует случайные данные о готовке, включая информацию о сезоне, типе блюда,
    основном ингредиенте, методе приготовления и результате.

    Returns:
    list: Список данных, каждая запись представляет собой информацию о готовке и выбор пользователя.
    """
    data = []
    for i in range(30):
        # Генерируем случайные данные для различных параметров приготовления
        season = randint(0, 1)  # Сезон (0 - зима, 1 - лето)
        dish_type = randint(0, 1)  # Тип блюда (0 - закуска, 1 - основное)
        main_ingredient = randint(0, 1)  # Основной ингредиент (0 - овощи, 1 - мясо)
        cooking_method = randint(0, 1)  # Способ приготовления (0 - жарка, 1 - запекание)
        difficulty_level = randint(0, 1)  # Степень сложности (0 - легко, 1 - сложно)
        result = randint(0, 1)  # Результат приготовления (0 - неуспешно, 1 - успешно)
        cooking_time = randint(1, 5)  # Время приготовления в минутах (1-5)

        # Выводим данные для понимания
        print(f"Сезон: {season}, Тип блюда: {dish_type}, Основной ингредиент: {main_ingredient}, "
              f"Способ приготовления: {cooking_method}, Степень сложности: {difficulty_level}, "
              f"Результат: {result}, Время приготовления: {cooking_time} минут")

        # Считываем, будет ли пользователь кушать это блюдо
        while True:
            try:
                eat_choice = int(input("Будешь ли ты кушать это блюдо? (0 - нет, 1 - да): "))
                if eat_choice in [0, 1]:
                    break
                else:
                    print("Пожалуйста, введите 0 или 1.")
            except ValueError:
                print("Пожалуйста, введите корректное число (0 или 1).")

        # Создаем запись данных
        entry = [
            season,
            dish_type,
            main_ingredient,
            cooking_method,
            difficulty_level,
            result,
            cooking_time,
            eat_choice
        ]

        data.append(entry)  # Добавляем запись в список данных

    return data  # Возвращаем сгенерированные данные

"""
Значения, генерируемые в функции:
- season: (0 - зима, 1 - лето)
- dish_type: (0 - закуска, 1 - основное)
- main_ingredient: (0 - овощи, 1 - мясо)
- cooking_method: (0 - жарка, 1 - запекание)
- difficulty_level: (0 - легко, 1 - сложно)
- result: (0 - неуспешно, 1 - успешно)
- cooking_time: время приготовления в минутах (от 1 до 5)
- eat_choice: (0 - нет, 1 - да)
"""

def regression(X, y):
    """
    Вычисляет коэффициенты линейной регрессии по заданным данным.

    Parameters:
    X (list): Список входных данных (признаки).
    y (list): Список меток (целевая переменная).

    Returns:
    ndarray: Коэффициенты регрессии.
    """
    n = len(y)  # Количество наблюдений
    # Пример: y = [50, 60, 80, 90], тогда n = 4

    M = []  # Матрица для уравнения регрессии
    b = []  # Вектор свободных членов

    # Заполняем первую строку матрицы M и первый элемент вектора b
    M.append([sum(x) for x in X] + [n])  # Сумма признаков + количество
    # Пример:
    # X = [[2, 3, 5, 7],
    #      [8, 7, 6, 5]]
    # Первая строка матрицы:
    # sum([2, 3, 5, 7]) = 17, sum([8, 7, 6, 5]) = 26, добавляем n = 4
    # M[0] = [17, 26, 4]

    b.append(sum(y))  # Сумма меток
    # Пример: y = [50, 60, 80, 90]
    # sum([50, 60, 80, 90]) = 280
    # b[0] = 280

    # Заполняем остальные строки матрицы M и вектор b
    for _, xl in enumerate(X):
        M.append([dot(x, xl) for x in X] + [sum(xl)])  # Векторные произведения
        # Пример:
        # Для x_1 = [2, 3, 5, 7]:
        # dot(x_1, x_1) = 2*2 + 3*3 + 5*5 + 7*7 = 87
        # dot(x_1, x_2) = 2*8 + 3*7 + 5*6 + 7*5 = 98
        # sum(x_1) = 17
        # Добавляем строку [87, 98, 17] в M
        # M[1] = [87, 98, 17]

        # Для x_2 = [8, 7, 6, 5]:
        # dot(x_2, x_1) = 98 (уже посчитано выше)
        # dot(x_2, x_2) = 8*8 + 7*7 + 6*6 + 5*5 = 174
        # sum(x_2) = 26
        # Добавляем строку [98, 174, 26] в M
        # M[2] = [98, 174, 26]

        b.append(dot(y, xl))  # Произведение меток и текущего вектора признаков
        # Пример:
        # Для x_1: dot(y, x_1) = 50*2 + 60*3 + 80*5 + 90*7 = 1380
        # Добавляем вектор: b[1] = 1380

        # Для x_2: dot(y, x_2) = 50*8 + 60*7 + 80*6 + 90*5 = 1640
        # Добавляем вектор: b[2] = 1640

    # Вычисляем коэффициенты регрессии методом Гаусса
    beta = gauss_slae(M, b)

    return beta  # Возвращаем коэффициенты регрессии

def generate_cooking_data() -> list:
    # Функция для генерации статических данных о приготовлении
    data = [
        [1, 1, 0, 1, 0, 0, 3, 1], [1, 0, 1, 1, 1, 0, 5, 1],
        [1, 1, 1, 0, 0, 0, 3, 1], [1, 1, 0, 1, 1, 0, 4, 1],
        [1, 1, 1, 0, 1, 0, 4, 1], [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 5, 1], [0, 1, 0, 0, 0, 0, 3, 0],
        [1, 0, 0, 0, 1, 0, 2, 0], [0, 1, 0, 0, 1, 1, 3, 1],
        [1, 0, 1, 1, 1, 1, 4, 1], [0, 1, 0, 1, 0, 0, 5, 0],
        [1, 1, 1, 1, 1, 0, 1, 0], [0, 0, 0, 1, 0, 0, 2, 0],
        [0, 1, 0, 1, 1, 0, 2, 0], [1, 1, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 1, 5, 1], [1, 0, 0, 0, 1, 1, 3, 1],
        [1, 0, 1, 0, 0, 0, 4, 1], [0, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 5, 0], [0, 1, 1, 0, 1, 1, 4, 1],
        [1, 1, 1, 1, 1, 1, 5, 1], [0, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 5, 1], [0, 1, 0, 1, 0, 1, 3, 1],
        [0, 0, 0, 1, 0, 1, 5, 1], [1, 0, 0, 0, 1, 1, 3, 1],
        [0, 1, 1, 0, 0, 1, 4, 1], [0, 0, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 1, 0, 1, 3, 1], [0, 0, 0, 0, 0, 0, 5, 0]
    ]
    return data  # Возвращаем статические данные

def main():
    """
    Основная функция программы: генерирует данные, подготавливает их для регрессии,
    вычисляет коэффициенты регрессии и оценивает модель на тестовых данных.
    """
    # Генерация данных о готовке
    cooking_data = generate_cooking_data()

    # Подготовка данных: создаем списки для признаков и меток
    features = [[] for _ in range(7)]  # Создаем список для каждого признака (метки мы здесь не храним)
    labels = [entry[-1] for entry in cooking_data[:-10]]  # Метки (целевая переменная), исключая последние 10 записей

    # Масштабируем данные
    scaled_data = scale([entry[:-1] for entry in cooking_data])  # Масштабируем признаки
    print(f'Результат шкалирования признаков: {scaled_data}')

    # Переструктурируем данные для регрессии
    for i in range(len(labels)):
        # Преобразуем метки в вероятности
        if labels[i] == 1:
            labels[i] = 0.95
        else:
            labels[i] = 0.05
        for j in range(7):
            features[j].append(scaled_data[i][j])  # Добавляем масштабированные признаки

    summa = 0
    for i in labels[:-10]:
        if i == 0.95:
            summa += 1

    print(summa)

    # Вычисляем коэффициенты регрессии
    coefficients = regression(features, labels)

    print(coefficients)

    # Тестируем модель на последних 10 записях данных
    true_pos, false_pos, false_neg, true_neg = 0, 0, 0, 0
    for i in range(20, 30):  # Используем последние 10 данных для теста
        prediction = mult_predict(scaled_data[i][:-1], coefficients)  # Делаем предсказание
        answer = round(exp(prediction) / (1 + exp(prediction)))  # Преобразуем предсказание в бинарный ответ (0 или 1)
        cor_answer = cooking_data[i][-1]  # Ожидаемый результат

        # Печать ошибки предсказания, если есть несоответствие
        if cor_answer != answer:
            print(f"Ошибка предсказания: ожидалось {cor_answer}, получено {answer}, данные: {cooking_data[i]}")

        # Подсчет метрик
        match answer, cor_answer:
            case 1, 1:
                true_pos += 1
            case 1, 0:
                false_pos += 1
            case 0, 1:
                false_neg += 1
            case 0, 0:
                true_neg += 1

    # Выводим метрики
    print(f"True Positive: {true_pos}, False Positive: {false_pos}, True Negative: {true_neg}, False Negative: {false_neg}")
    print(f"F1-метрика: {f1_score(true_pos, false_pos, false_neg)}")  # Выводим F1-метрику

if __name__ == "__main__":
    main()
