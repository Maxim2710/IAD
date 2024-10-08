# Функция для извлечения комплексных чисел из двумерного списка
def extract_complex_numbers(two_level_list):
    # Список для хранения найденных комплексных чисел
    complex_numbers = []

    # Проходим по каждому подсписку в двумерном списке
    for sublist in two_level_list:
        # Проходим по каждому элементу в подсписке
        for item in sublist:
            # Если элемент является комплексным числом, добавляем его в список
            if isinstance(item, complex):
                complex_numbers.append(item)

    # Возвращаем найденные комплексные числа в виде кортежа
    return tuple(complex_numbers)


# Пример двумерного списка с различными типами данных, включая комплексные числа
two_level_list = [
    [1, 2.5, 3 + 8j, 5],  # Подсписок с комплексным числом 3+8j
    [6.7, 2 + 3j, 4, 5 + 1j],  # Подсписок с комплексными числами 2+3j и 5+1j
    [-1, 0.5, 7, 1j]  # Подсписок с комплексным числом 1j
]

# Вызов функции для извлечения комплексных чисел из списка
result = extract_complex_numbers(two_level_list)

# Вывод результатов
print("---------------------------------------")
print("Задание 6: Выбор комплексных чисел и запись их в кортеж")
print("Кортеж комплексных чисел:", result)
print("---------------------------------------")
