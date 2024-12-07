from collections import defaultdict
import random
from nltk.tokenize import word_tokenize

def build_ngram_model(text: str, n: int) -> tuple:
    """
    Создает модель n-грамм на основе текста.

    Parameters
    ----------
    text : str
        Текст для обработки.
    n : int
        Размер n-граммы.

    Returns
    -------
    tuple
        Кортеж с инициализирующими фразами и словарем переходов n-грамм.
    """
    words = word_tokenize(text)  # Разбиваем текст на слова.
    ngrams = zip(*[words[i:] for i in range(n)]) # последовательности длиной n
    trans = defaultdict(list)
    init = []

    for ngram in ngrams:
        # Если текущая n-грамма начинается с точки, добавляем начальную последовательность.
        if ngram[0] == '.':
            init.append(ngram[1:n - 1])
        # Добавляем следующую последовательность в словарь переходов.
        trans[ngram[0:n - 1]].append(ngram[-1])

    return init, trans


def generate_text(init: list, trans: defaultdict, n: int) -> str:
    """
    Генерирует текст на основе модели n-грамм.

    Parameters
    ----------
    init : list
        Список начальных последовательностей.
    trans : defaultdict
        Словарь переходов n-грамм.
    n : int
        Размер n-граммы.

    Returns
    -------
    str
        Сгенерированный текст.
    """
    result = random.choice(init)  # Начальная последовательность.
    fr = ['.'] # контекст для формирования следующей последовательности
    res_true = [result[0]]

    while True:
        # Находим кандидатов для следующего слова.
        candidates = trans[tuple(fr + list(result))]
        next_word = random.choice(candidates)

        # Обновляем контекст для следующего слова.
        fr = [result[0]]
        result = list(result[1:]) + [next_word]
        res_true.append(result[0])

        # Завершаем генерацию текста, если встречаем точку.
        if result[0] == '.':
            return " ".join(res_true)


def main() -> None:
    """
    Основная функция программы. Выводит сгенерированные тексты на основе триграмм и четырёхграмм.
    """
    file_path = "../data/data_scientist_text.txt"  # Имя файла для обработки.

    # Загружаем текст из файла.
    with open(file_path, "r", encoding="UTF8") as f:
        text = f.read()

    # Обрабатываем текст для триграмм и четырёхграмм.
    for n in (3, 4):
        init, trans = build_ngram_model(text, n)  # Создаем модель n-грамм.
        print(f"{n}-gram: {generate_text(init, trans, n)}")


if __name__ == "__main__":
    main()
