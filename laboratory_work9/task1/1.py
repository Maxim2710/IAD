import csv
from dsmltf import scale, KMeans, generate_clusters, get_children, distance, is_leaf, squared_distance
import matplotlib.pyplot as plt


# def load_and_process_data(file_path: str) -> list:
#     """
#     Загружает данные из CSV файла, обрабатывает их и возвращает список выбранных колонок.
#
#     Parameters
#     ----------
#     file_path : str
#         Путь к CSV файлу, содержащему данные.
#
#     Returns
#     -------
#     list
#         Список обработанных данных, включающий закодированные жанры, продолжительность, IMDB рейтинг и язык.
#     """
#     data = []  # Список для хранения обработанных данных
#     genre_map = {}  # Словарь для хранения уникальных жанров и их индексов
#     language_map = {}  # Словарь для хранения уникальных языков и их индексов
#     genre_index = 0  # Индекс для жанров
#     language_index = 0  # Индекс для языков
#
#     # Открываем CSV файл и начинаем читать его
#     with open(file_path, "r", encoding="UTF-8") as f:
#         reader = csv.reader(f)  # Используем csv.reader для чтения файла
#         headers = next(reader)  # Пропускаем строку с заголовками
#
#         # Проходим по всем строкам данных
#         for row in reader:
#             genre = row[1]  # Извлекаем жанр из строки
#             if genre not in genre_map:
#                 genre_map[genre] = genre_index  # Если жанр новый, добавляем его в словарь
#                 genre_index += 1  # Увеличиваем индекс для следующего жанра
#
#             language = row[5]  # Извлекаем язык из строки
#             if language not in language_map:
#                 language_map[language] = language_index  # Если язык новый, добавляем его в словарь
#                 language_index += 1  # Увеличиваем индекс для следующего языка
#
#             # Кодируем жанр и язык в числа, а также сохраняем продолжительность и IMDB рейтинг
#             genre_encoded = genre_map[genre]
#             runtime = int(row[3])  # Продолжительность
#             imdb_score = float(row[4])  # IMDB рейтинг
#             language_encoded = language_map[language]
#
#             # Добавляем обработанные данные в список
#             data.append([genre_encoded, runtime, imdb_score, language_encoded])
#
#     return data  # Возвращаем список обработанных данных

def load_and_process_data(file_path: str) -> list:
    """
    Загружает данные из CSV файла, обрабатывает их и возвращает список с продолжительностью и рейтингом IMDB.

    Parameters
    ----------
    file_path : str
        Путь к CSV файлу, содержащему данные.

    Returns
    -------
    list
        Список обработанных данных, включающий продолжительность и IMDB рейтинг.
    """
    data = []  # Список для хранения обработанных данных

    # Открываем CSV файл и начинаем читать его
    with open(file_path, "r", encoding="UTF-8") as f:
        reader = csv.reader(f)  # Используем csv.reader для чтения файла
        headers = next(reader)  # Пропускаем строку с заголовками

        # Проходим по всем строкам данных
        for row in reader:
            runtime = int(row[3])  # Извлекаем продолжительность
            imdb_score = float(row[4])  # Извлекаем IMDB рейтинг
            # Добавляем обработанные данные в список
            data.append([runtime, imdb_score])

    return data  # Возвращаем список обработанных данных

def squared_errors(inps, k):
    """
    Вычисляет сумму квадратов ошибок для заданного количества кластеров.

    Parameters
    ----------
    inps : list
        Список данных для кластеризации.
    k : int
        Количество кластеров.

    Returns
    -------
    float
        Сумма квадратов ошибок.
    """
    # Создаем объект класса KMeans с заданным числом кластеров
    clasterbuilder = KMeans(k)

    # Обучаем модель KMeans на переданных данных
    clasterbuilder.train(inps)

    # Получаем центроиды (средние точки) для каждого из кластеров
    means = clasterbuilder.means

    # Классифицируем каждую точку, присваивая её к ближайшему кластеру
    inclaster = map(clasterbuilder.classify, inps)

    # Суммируем квадраты расстояний между каждой точкой и её центроидом
    return sum(squared_distance(inp, means[cluster])  # Рассчитываем квадрат расстояния для каждой точки
               for inp, cluster in zip(inps, inclaster))  # Проходим по всем точкам и их кластерам

def plot_squared_errors(x, y):
    """
    Строит график зависимости квадратных ошибок от числа кластеров.

    Parameters
    ----------
    x : list
        Список значений количества кластеров.
    y : list
        Список значений квадратных ошибок.

    Returns
    -------
    None
        График не возвращает значений, а только отображается.
    """
    plt.figure(figsize=(10, 5))  # Настроим размер графика
    plt.plot(x, y, label='Squared Error vs Number of Clusters', marker='o', color='b')  # Строим график
    plt.title('Squared Error for Different Numbers of Clusters', fontsize=16)  # Заголовок графика
    plt.xlabel('Number of Clusters (k)', fontsize=12)  # Метка оси X
    plt.ylabel('Squared Error', fontsize=12)  # Метка оси Y
    plt.grid(True)  # Включаем сетку для удобства чтения графика
    plt.legend()  # Добавляем легенду
    plt.show()  # Отображаем график

def plot_clusters(data, kmeans, xlim=None, ylim=None):
    """
    Визуализирует точки данных с их кластерами.

    Parameters
    ----------
    data : list
        Список данных для визуализации.
    kmeans : KMeans
        Обученная модель KMeans.
    xlim : tuple, optional
        Лимиты для оси X (min, max).
    ylim : tuple, optional
        Лимиты для оси Y (min, max).

    Returns
    -------
    None
        График не возвращает значений, а только отображается.
    """
    # Получаем предсказания кластеров для каждого элемента данных
    clusters = [kmeans.classify(point) for point in data]

    plt.figure(figsize=(10, 6))  # Устанавливаем размер графика
    for i, cluster in enumerate(set(clusters)):
        # Отбираем точки, принадлежащие текущему кластеру
        cluster_points = [point for j, point in enumerate(data) if clusters[j] == cluster]
        cluster_points = list(zip(*cluster_points))  # Разделяем координаты для удобства

        # Рисуем точки, относящиеся к текущему кластеру
        plt.scatter(cluster_points[0], cluster_points[1], label=f"Cluster {i + 1}", s=50)

    # Устанавливаем лимиты осей, если они заданы
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    # Добавляем подписи осей и легенду
    plt.title("Cluster Visualization", fontsize=16)
    plt.xlabel("Scaled Runtime", fontsize=12)
    plt.ylabel("Scaled IMDB Score", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

def train_kmeans_model(data, k):
    """
    Обучает модель KMeans с заданным количеством кластеров.

    Parameters
    ----------
    data : list
        Список данных для кластеризации.
    k : int
        Количество кластеров.

    Returns
    -------
    KMeans
        Обученная модель KMeans.
    """
    kmeans = KMeans(k)  # Создаем модель KMeans с k кластерами
    kmeans.train(data)  # Обучаем модель на данных
    return kmeans  # Возвращаем обученную модель

def main():
    """
    Основная функция для кластеризации данных и отображения графика.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Выполняет кластеризацию, строит график и выводит результаты.
    """
    data_set = load_and_process_data("../data/NetflixOriginals.csv")  # Загружаем и обрабатываем данные из файла
    scaled_data = scale(data_set[:100])  # Масштабируем данные (используем только первые 100 строк)

    # scaled_data = scale(data_set)
    # x = []
    # y = []
    # for k in range(1, 20):
    #     x.append(k)
    #     y.append(squared_errors(scaled_data, k))
    # print(x)
    # print(y)

    # Предварительно вычисленные квадраты ошибок для разных значений k
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    y = [2335.999999999999, 2121.8352144649193, 1733.7098159551483, 1731.3833543466562, 1280.8200011472968,
         1322.1644570532922, 1252.6341500009034, 1298.8260055309267, 947.215995282551, 1128.3021378553422,
         1069.8306918750236, 743.2111211305224, 793.3788479920217, 918.4788661897383, 694.7464225286514,
         793.8435582134562, 1007.5325400497563, 694.594807606682, 696.2283658958489]

    plot_squared_errors(x, y)  # Строим график зависимости квадратных ошибок от числа кластеров

    optimal_k = 5

    kmeans = train_kmeans_model(scaled_data, optimal_k)  # Обучаем модель KMeans с 9 кластерами
    print(kmeans.means)  # Выводим центры кластеров

    # Иерархическая кластеризация
    base_cluster = bottom_up_cluster(scaled_data)  # Строим иерархическую кластеризацию
    print([get_values(cluster) for cluster in generate_clusters(base_cluster, optimal_k)])  # Выводим кластеры

    plot_clusters(scaled_data, kmeans, xlim=(-6, 4), ylim=(-5, 3))

def bottom_up_cluster(inputs, distance_agg=min):
    """
    Выполняет иерархическую кластеризацию данных методом "снизу-вверх".

    Parameters
    ----------
    inputs : list
        Список данных для кластеризации.
    distance_agg : function, optional
        Функция для агрегации расстояний (по умолчанию используется минимальное расстояние).

    Returns
    -------
    tuple
        Иерархически скомпонованный кластер.
    """
    clusters = [(inp,) for inp in inputs]  # Создаем начальные кластеры (каждое значение в отдельном кластере)
    while len(clusters) > 1:  # Пока в списке кластеров больше одного
        # Находим пару кластеров с минимальным расстоянием
        c1, c2 = min([(cluster1, cluster2)
                      for i, cluster1 in enumerate(clusters)
                      for cluster2 in clusters[:i]],
                     key=lambda x: cluster_distance(*x, distance_agg))
        clusters = [c for c in clusters if c != c1 and c != c2]  # Удаляем эти два кластера из списка
        merged_cluster = (len(clusters), [c1, c2])  # Объединяем два кластера в один
        clusters.append(merged_cluster)  # Добавляем новый объединенный кластер в список

    return clusters[0]  # Возвращаем финальный кластер


def get_values(cluster):
    """
    Извлекает значения из кластера.

    Parameters
    ----------
    cluster : tuple
        Кластер, из которого необходимо извлечь значения.

    Returns
    -------
    list
        Список значений из кластера.
    """
    if is_leaf(cluster):  # Если кластер является "листьем" (он не содержит другие кластеры)
        return [cluster[0]]  # Возвращаем сам элемент
    else:
        # Рекурсивно извлекаем значения из всех вложенных кластеров
        return [val for child in get_children(cluster) for val in get_values(child)]


def cluster_distance(cluster1, cluster2, distance_agg=min):
    """
    Вычисляет расстояние между двумя кластерами.

    Parameters
    ----------
    cluster1 : tuple
        Первый кластер.
    cluster2 : tuple
        Второй кластер.
    distance_agg : function, optional
        Функция для агрегации расстояний (по умолчанию используется минимальное расстояние).

    Returns
    -------
    float
        Расстояние между двумя кластерами.
    """
    values1 = list(get_values(cluster1))  # Извлекаем все значения из первого кластера
    values2 = list(get_values(cluster2))  # Извлекаем все значения из второго кластера
    # Вычисляем расстояние между всеми парами значений из двух кластеров и агрегация их с помощью distance_agg
    return distance_agg([distance(list(inp1), list(inp2)) for inp1 in values1 for inp2 in values2])


if __name__ == "__main__":
    main()  # Запускаем основную функцию

