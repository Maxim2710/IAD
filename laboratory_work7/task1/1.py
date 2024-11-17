import csv
import ssl
import nltk
from dsmltf import count_words, spam_probability, f1_score
from nltk import pos_tag
from collections import defaultdict

# Обход SSL-проверки для загрузки ресурса NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Загрузка правильного ресурса для POS-теггера
nltk.download("averaged_perceptron_tagger")


def word_probabilities(word_counts: list[tuple], total_spams: int, total_non_spams: int,
                       k=0.5) -> list[tuple]:
    """
    Вычисляет вероятности слов в спам- и не-спам-сообщениях с использованием сглаживания Лапласа.

    Parameters
    ----------
        word_counts (list[tuple]): Список кортежей, содержащих количество слов.
        total_spams (int): Общее количество спам-сообщений.
        total_non_spams (int): Общее количество не-спам-сообщений.
        k (float): Параметр сглаживания для вероятностей.

    Returns
    -------
        list[tuple]: Список кортежей, содержащих слово и его вероятности в спам- и не-спам-сообщениях.
    """
    return [(word[0], (word[1] + k) / (total_spams + 2 * k),
             (word[2] + k) / (total_non_spams + 2 * k))
            for word in word_counts]


def load_data_from_csv(file_path: str) -> list:
    """
    Загружает данные из CSV-файла и преобразует их в список сообщений и меток.

    Parameters
    ----------
        file_path (str): Путь к CSV-файлу с данными.

    Returns
    -------
        list: Список кортежей, где каждый кортеж содержит сообщение и его метку (1 для спама, 0 для не-спама).
    """
    with open(file_path) as f:
        data = []
        for row in csv.reader(f):
            data.append([row[1], 1 if row[0] == "spam" else 0])
    return data


def evaluate_model(words: list[tuple], test_data: list) -> float:
    """
    Оценивает модель, сравнивая предсказанные значения с фактическими.

    Parameters
    ----------
        words (list[tuple]): Список кортежей слов и их вероятностей.
        test_data (list): Список тестовых данных для оценки.

    Returns
    -------
        float: Значение F1-меры для оценки модели.
    """
    true_pos, false_pos, false_neg = 0, 0, 0
    for message in test_data:
        predicted = round(spam_probability(words, message[0]))
        actual = message[1]
        if predicted == 1 and actual == 1:
            true_pos += 1
        elif predicted == 1 and actual == 0:
            false_pos += 1
        elif predicted == 0 and actual == 1:
            false_neg += 1

    return f1_score(true_pos, false_pos, false_neg)


def prepare_training_data(dataset: list) -> tuple:
    """
    Подготавливает обучающие данные, подсчитывая количество спам- и не-спам-сообщений
    и фильтруя прилагательные из обучающего набора.

    Parameters
    ----------
        dataset (list): Список сообщений и их меток для подготовки.

    Returns
    -------
        tuple: Кортеж, содержащий:
            - Словарь, где ключи - слова, а значения - списки количеств спама и не-спама.
            - Общее количество спам-сообщений.
            - Общее количество не-спам-сообщений.
    """
    spam_count = len([msg for msg in dataset if msg[1] == 1])
    ham_count = len(dataset) - spam_count
    train_set = count_words(dataset)

    # Фильтрация прилагательных из обучающего набора
    tagged_words = pos_tag(train_set.keys())
    filtered_train_set = defaultdict(lambda: [0, 0], {key: train_set[key] for key, tag in tagged_words if
                                                      tag not in ('JJ', 'JJR', 'JJS')})

    return filtered_train_set, spam_count, ham_count


def main() -> None:
    """
    Главная функция для загрузки данных, подготовки обучающих данных,
    оценки модели и вывода результатов.

    Returns
    -------
        None
    """
    dataset = load_data_from_csv("../data/spam.csv")

    # Ручное разделение данных
    split_index = int(len(dataset) * 0.8)
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]

    # Подготовка обучающих данных
    train_set, spam_count, ham_count = prepare_training_data(train_data)

    # Получение самых частых слов в спаме
    frequent_words = sorted(train_set, key=lambda w: train_set[w][0] if len(w) >= 4 else 0)[-40:]
    words = [(word, train_set[word][0] / spam_count,
              train_set[word][1] / ham_count if train_set[word][1] / ham_count else 0.01) for word in frequent_words]

    # Оценка без сглаживания
    print("Без сглаживания:", evaluate_model(words, test_data))

    # Применение сглаживания и оценка
    smoothed_words = word_probabilities(
        [(word[0], train_set[word[0]][0], train_set[word[0]][1]) for word in words], spam_count, ham_count)

    print("Со сглаживанием:", evaluate_model(smoothed_words, test_data))


if __name__ == "__main__":
    main()
