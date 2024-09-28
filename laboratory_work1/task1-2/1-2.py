# Функция для проверки, является ли число четным
def is_even(n):
    return (n & 1) == 0  # Операция побитового "И" с единицей: если последний бит 0, то число четное

# Функция для деления числа на два с использованием битового сдвига вправо
def divide_by_two(n):
    return n >> 1  # Сдвиг битов вправо эквивалентен делению числа на 2

# Функция для умножения числа на два с использованием битового сдвига влево
def mult_by_two(n):
    return n << 1  # Сдвиг битов влево эквивалентен умножению числа на 2

# Функция для нахождения наибольшего общего делителя (НОД) двух чисел, используя модифицированный алгоритм Евклида
def gcd(a, b):
    # Если числа равны, их НОД равен одному из них
    if a == b:
        return a

    # Если оба числа четные, делим их на два и умножаем результат на два
    elif is_even(a) and is_even(b):
        return mult_by_two(gcd(divide_by_two(a), divide_by_two(b)))

    # Если первое число четное, делим его на два и продолжаем вычисление НОД
    elif is_even(a):
        return gcd(divide_by_two(a), b)

    # Если второе число четное, делим его на два и продолжаем вычисление НОД
    elif is_even(b):
        return gcd(divide_by_two(b), a)

    # Если оба числа нечетные и первое больше второго, вычитаем меньшее из большего и продолжаем вычисление НОД
    elif a > b:
        return gcd(divide_by_two(a - b), b)
    else:
        # Если второе число больше первого, аналогично вычитаем и продолжаем
        return gcd(a, divide_by_two(b - a))

# Функция для получения корректного 10-значного числа от пользователя
def get_valid_number():
    while True:
        try:
            # Запрашиваем ввод числа у пользователя
            n = int(input("Введите 10-значное число: "))
            # Проверяем, является ли число 10-значным
            if n < 1000000000 or n > 9999999999:
                raise ValueError("Число должно быть 10-значным.")
            return n
        except ValueError as e:
            # Обрабатываем ошибку ввода, если она произошла, и просим ввести число снова
            print(f"Ошибка: {e}. Пожалуйста, попробуйте снова.")

# Основная программа
print("---------------------------------------")

# Получаем корректное 10-значное число от пользователя
n = get_valid_number()

# Определяем константное число для вычисления НОД
CONST_NUM = 8

# Вычисляем НОД числа 8 и введенного пользователем 10-значного числа
result = gcd(CONST_NUM, n)

# Выводим результат
print("Задание 1-2: Нахождение НОД числа 8 и 10-значного числа/обработка некорректного ввода")
print(f"НОД числа 8 и {n} равен {result}")
print("---------------------------------------")
