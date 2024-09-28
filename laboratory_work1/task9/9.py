from math import gcd  # Импортируем функцию gcd для нахождения наибольшего общего делителя


# Класс для работы с дробями
class Frac:
    # Конструктор класса, принимающий числитель и знаменатель
    def __init__(self, numerator, denominator):
        if denominator == 0:  # Проверка, чтобы знаменатель не был равен нулю
            raise ValueError("Знаменатель не может быть нулем")
        self.numerator = numerator  # Устанавливаем числитель
        self.denominator = denominator  # Устанавливаем знаменатель
        self.simplify()  # Сразу упрощаем дробь после создания

    # Метод для упрощения дроби с использованием НОД (gcd)
    def simplify(self):
        common_divisor = gcd(self.numerator, self.denominator)  # Находим НОД числителя и знаменателя
        # Делим числитель и знаменатель на их общий делитель
        self.numerator //= common_divisor
        self.denominator //= common_divisor
        if self.denominator < 0:  # Если знаменатель отрицательный, делаем его положительным
            self.numerator = -self.numerator  # Меняем знак числителя
            self.denominator = -self.denominator  # Меняем знак знаменателя

    # Метод для строкового представления дроби
    def __repr__(self):
        return f"{self.numerator}/{self.denominator}"  # Возвращаем строку вида 'числитель/знаменатель'

    # Метод для обращения дроби
    def inverse(self):
        if self.numerator == 0:  # Если числитель равен нулю, дробь нельзя обратить
            raise ZeroDivisionError("Нельзя обратить дробь с числителем равным 0")
        return Frac(self.denominator,
                    self.numerator)  # Возвращаем новую дробь с перевернутыми числителем и знаменателем

    # Метод для сложения двух дробей
    def __add__(self, other):
        # Формула сложения дробей: (a/b) + (c/d) = (a*d + c*b) / (b*d)
        new_numerator = self.numerator * other.denominator + other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        return Frac(new_numerator, new_denominator)  # Возвращаем результат сложения как новую дробь

    # Метод для умножения двух дробей
    def __mul__(self, other):
        # Формула умножения дробей: (a/b) * (c/d) = (a*c) / (b*d)
        return Frac(self.numerator * other.numerator, self.denominator * other.denominator)


# Примеры использования класса Frac

frac1 = Frac(3, 5)  # Создаем дробь 3/5
frac2 = Frac(1, 4)  # Создаем дробь 1/4

# Вывод результатов
print("---------------------------------------")
print("Задание 9: Методы сложения, умножения и обращения с дробью")
print("Дробь 1:", frac1)  # Выводим первую дробь
print("Дробь 2:", frac2)  # Выводим вторую дробь
print("Обратная дробь 1:", frac1.inverse())  # Выводим обратную дробь к первой
print("Сумма дробей:", frac1 + frac2)  # Выводим результат сложения дробей
print("Произведение дробей:", frac1 * frac2)  # Выводим результат умножения дробей
print("---------------------------------------")
