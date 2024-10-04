from random import randint
from numpy.linalg import solve as gauss
import numpy as np
from math import exp
from dsmltf import scale, mult_r_squared, mult_predict, dot

def make_cooking_data() -> list:
    data = []
    for i in range(30):
        # Генерируем данные
        season = randint(0, 1)  # Сезон (0 - зима, 1 - лето)
        dish_type = randint(0, 1)  # Тип блюда (0 - закуска, 1 - основное)
        main_ingredient = randint(0, 1)  # Основной ингредиент (0 - овощи, 1 - мясо)
        cooking_method = randint(0, 1)  # Способ приготовления (0 - жарка, 1 - запекание)
        difficulty_level = randint(0, 1)  # Степень сложности (0 - легко, 1 - сложно)
        result = randint(0, 1)  # Результат (0 - неуспешно, 1 - успешно)
        cooking_time = randint(1, 5)  # Время приготовления в минутах (1-5)

        # Печатаем данные для понимания
        print(f"Сезон: {season}, Тип блюда: {dish_type}, Основной ингредиент: {main_ingredient}, "
              f"Способ приготовления: {cooking_method}, Степень сложности: {difficulty_level}, "
              f"Результат: {result}, Время приготовления: {cooking_time} минут")

        # Считываем, будете ли вы кушать это блюдо
        while True:
            try:
                eat_choice = int(input("Будешь ли ты кушать это блюдо? (0 - нет, 1 - да): "))
                if eat_choice in [0, 1]:
                    break
                else:
                    print("Пожалуйста, введите 0 или 1.")
            except ValueError:
                print("Пожалуйста, введите корректное число (0 или 1).")

        # Создаем запись в виде списка
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

def regression(X, y):  # Функция для вычисления коэффициентов регрессии
    n = len(y)  # Количество наблюдений
    M = []  # Матрица для регрессии
    b = []  # Вектор для правой части уравнения

    # Добавляем первую строку в матрицу M и элемент в вектор b
    M.append([sum(x) for x in X] + [n])  # Суммы по столбцам + количество
    b.append(sum(y))  # Сумма по целевой переменной

    # Заполняем матрицу M и вектор b для каждого наблюдения
    for _, xl in enumerate(X):
        M.append([dot(x, xl) for x in X] + [sum(xl)])  # Векторные произведения
        b.append(dot(y, xl))  # Произведение целевой переменной и текущего вектора

    # Вычисляем коэффициенты регрессии с помощью метода Гаусса
    beta = gauss(np.array(M, dtype="float64"), np.array(b, dtype="float64"))
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
    # Генерация данных о приготовлении
    cooking_data = generate_cooking_data()

    # Подготовка данных для обработки
    features = [[] for _ in range(7)]  # Список для признаков
    labels = []  # Список для меток (результатов)

    # Масштабируем данные
    scaled_data = scale(cooking_data)

    # Переструктурируем данные для регрессии
    for i in range(len(scaled_data) - 10):
        for j in range(7):
            features[j].append(scaled_data[i][j])  # Заполняем признаки
        labels.append(scaled_data[i][-1])  # Добавляем метку

    # Вычисляем коэффициенты регрессии
    coefficients = regression(features, labels)
    print(f"Коэффициенты регрессии: {coefficients}")

    # Вычисляем R-квадрат ошибки
    r_squared = mult_r_squared([item[:-1] for item in scaled_data[-10:]], labels, coefficients)

    # Тестируем на тестовом наборе данных
    for i in range(15, 30):
        prediction = mult_predict(scaled_data[i][:-1], coefficients)  # Делаем предсказание
        answer = 1 if exp(prediction) / (1 + exp(prediction)) > 0.5 else 0  # Преобразуем предсказание в бинарный ответ
        if cooking_data[i][-1] != answer:  # Проверяем правильность предсказания
            print(f"Ошибка предсказания: ожидалось {cooking_data[i][-1]}, получено {answer}, данные: {cooking_data[i]}")

    print(f"Ошибка R-квадрата: {r_squared}")  # Выводим R-квадрат ошибки

if __name__ == "__main__":
    main()  # Запускаем главную функцию
