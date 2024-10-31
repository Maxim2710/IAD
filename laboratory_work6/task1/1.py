import csv
from dsmltf import knn_classify, k_neighbours_classify

def load_earthquake_data(file_path: str) -> list:
    """
    Загрузка данных о землетрясениях из CSV файла.

    Parameters
    ----------
        file_path (str): Путь к CSV файлу с данными о землетрясениях.

    Returns
    -------
        list: Список данных, содержащий широту, долготу и магнитуду.
    """
    data = []
    with open(file_path, "r") as csv_file:  # Открытие файла для чтения
        for row in csv.reader(csv_file, delimiter=","):  # Чтение данных по строкам
            try:
                # Добавляем только широту, долготу и магнитуду
                data.append([float(row[1]), float(row[2]), float(row[4])])
            except:
                continue
    return data  # Возвращаем список данных

def is_within_bounds(latitude: float, longitude: float) -> bool:
    """
    Проверка координат, находятся ли они в пределах допустимого диапазона.

    Parameters
    ----------
        latitude (float): Широта точки.
        longitude (float): Долгота точки.

    Returns
    -------
        bool: True, если координаты находятся в допустимом диапазоне, иначе False.
    """
    # Диапазон широт и долгот для заданного региона
    # LAT_MIN, LAT_MAX = -5.703, 0.352
    # LON_MIN, LON_MAX = 112.764, 123.311
    LAT_MIN, LAT_MAX = 34.002, 39
    LON_MIN, LON_MAX = 136.52, 142

    if LAT_MIN <= latitude <= LAT_MAX and LON_MIN <= longitude <= LON_MAX:
        return True  # Если координаты в пределах допустимого диапазона
    else:
        # Выводим сообщение об ошибке и возвращаем False
        print(f"Ошибка: координаты [{latitude}, {longitude}] находятся вне допустимого диапазона!")
        return False

def find_best_k(classifications_results: dict) -> int:
    """
    Находит лучшее значение параметра k для классификации.

    Parameters
    ----------
        classifications_results (dict): Результаты классификации для различных значений k.

    Returns
    -------
        int: Оптимальное значение k с наибольшей точностью.
    """
    # Вычисляем точность для каждого значения k
    accuracy_per_k = [classifications_results[k][0] / classifications_results[k][1] for k in range(1, 12)]
    # Находим индекс с максимальной точностью и возвращаем значение k
    return accuracy_per_k.index(max(accuracy_per_k)) + 1

def run_classification(file_path: str):
    """
    Основная функция для запуска процесса классификации данных о землетрясениях.

    Parameters
    ----------
        file_path (str): Путь к CSV файлу с данными о землетрясениях.

    Returns
    -------
        None
    """
    # Загрузка данных о землетрясениях
    earthquake_data = load_earthquake_data(file_path)

    # Преобразование данных: оставляем широту и долготу, а магнитуду используем как метку
    data_with_exact_magnitude = [(data[:-1], data[-1]) for data in earthquake_data]
    # То же самое, но магнитуда округляется до ближайшего целого числа
    data_with_rounded_magnitude = [(data[:-1], round(data[-1])) for data in earthquake_data]

    # Классификация методом k-ближайших соседей для точной магнитуды
    classifications_results_exact = k_neighbours_classify(11, data_with_exact_magnitude[:200])
    # Классификация для округленной магнитуды
    classifications_results_rounded = k_neighbours_classify(11, data_with_rounded_magnitude[:200])

    # Поиск лучшего значения k для обоих наборов данных
    best_k_exact = find_best_k(classifications_results_exact)
    best_k_rounded = find_best_k(classifications_results_rounded)

    print(best_k_exact, best_k_rounded)
    try:
        # Ввод координат от пользователя
        latitude, longitude = map(float, input("Введите широту и долготу (допустимый диапазон: от 34.002 до 39 (с. ш.), от 136.52 до 142 (в. д.): ").split())
    except ValueError:
        # Ошибка, если введены некорректные значения
        print("Ошибка: Введите корректные числовые значения для широты и долготы.")
        return

    # Проверяем, находятся ли введённые координаты в допустимом диапазоне
    if is_within_bounds(latitude, longitude):
        # Выполняем классификацию для данных с точной магнитудой
        result_exact = knn_classify(best_k_exact, data_with_exact_magnitude, (latitude, longitude))
        # Выполняем классификацию для данных с округленной магнитудой
        result_rounded = knn_classify(best_k_rounded, data_with_rounded_magnitude, (latitude, longitude))

        # Выводим результаты классификации
        print(f"Классификация с точной магнитудой: {result_exact}")
        print(f"Классификация с округленной магнитудой: {result_rounded}")
    else:
        # Сообщение об ошибке, если координаты не попадают в допустимый диапазон
        print("Попробуйте снова с корректными координатами.")

if __name__ == "__main__":
    # Запуск классификации с использованием данных о землетрясениях в Центр. Японии
    run_classification("../data/earthquakes_japan.csv")
