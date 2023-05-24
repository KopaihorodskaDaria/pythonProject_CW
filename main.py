import genetic_algorithm as ga
import random
import numpy as np
import LR_alg
import experinents as exp
import Greedy_alg
import min_column_max_row as mm


def print_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print(matrix[i][j], end='')
        print()


def create_matrix_of_parlament(n, K_characteristik, k):
    matrix = np.zeros((n, k), dtype=int)
    # Ensure each column has at least one 1
    for j in range(k):
        while np.sum(matrix[:, j]) == 0:
            row_index = random.randint(0, n - 1)
            if np.sum(matrix[row_index, :]) < K_characteristik:
                matrix[row_index, j] = 1
    # Add K random 1s to each row
    for i in range(n):
        remaining_ones = K_characteristik - np.sum(matrix[i, :])
        while remaining_ones > 0:
            column_index = random.randint(0, k - 1)
            if matrix[i, column_index] == 0 and np.sum(matrix[:, column_index]) < n:
                matrix[i, column_index] = 1
                remaining_ones -= 1
    return matrix


def random_enter():
    number_of_parliamentarians = input('Введіть кількість парламентарів: \n >>>> ')
    number_of_characteristics = input('Введіть кількість характеристик парламентарів: \n >>>> ')
    number_of_sign = input('Введіть кількість ознак: \n >>>> ')
    # number_of_iteration = input('Enter number of iteration to stop: \n')
    print('Введені дані: \n ')
    print('Кількість парламентарів: ', number_of_parliamentarians)
    print('Кількість характеристик парламентарів: ', number_of_characteristics)
    print('Кількість ознак: ', number_of_sign)
    matrix_of_parlament = create_matrix_of_parlament(int(number_of_parliamentarians), int(number_of_characteristics),
                                                     int(number_of_sign))
    print("Згенерована матриця: ")
    print_matrix(matrix_of_parlament)
    return matrix_of_parlament, number_of_parliamentarians, number_of_characteristics, number_of_sign


def file_enter():
    path = input('Введіть шлях до файлу до файлу: \n >>>> ')
    matrix_of_parlament = np.genfromtxt(path, dtype='int', delimiter=' ')
    print_matrix(matrix_of_parlament)
    number_of_parliamentarians = len(matrix_of_parlament)
    number_of_characteristics = np.sum(matrix_of_parlament[0])
    number_of_sign = len(matrix_of_parlament[len(matrix_of_parlament) - 1])
    print('Введені дані: \n ')
    print('Кількість парламентарів: ', number_of_parliamentarians)
    print('Кількість характеристик парламентарів: ', number_of_characteristics)
    print('Кількість ознак: ', number_of_sign)
    return matrix_of_parlament, number_of_parliamentarians, number_of_characteristics, number_of_sign


def manually_enter():
    number_of_parliamentarians = input('Введіть кількість парламентарів: \n >>>> ')
    number_of_characteristics = input('Введіть кількість характеристик парламентарів: \n >>>> ')
    number_of_sign = input('Введіть кількість ознак: \n >>>> ')
    print('Введені дані: \n ')
    print('Кількість парламентарів: ', number_of_parliamentarians)
    print('Кількість характеристик парламентарів: ', number_of_characteristics)
    print('Кількість ознак: ', number_of_sign)
    print(f'Введіть матрицю розміром {number_of_parliamentarians}x{number_of_sign} \n')
    matrix_of_parlament = []
    for i in range(int(number_of_parliamentarians)):
        row = input().split()
        for i in range(len(row)):
            row[i] = int(row[i])
        matrix_of_parlament.append(row)
    print_matrix(matrix_of_parlament)
    return matrix_of_parlament, number_of_parliamentarians, number_of_characteristics, number_of_sign


def choose_algorithm(matrix_of_parlament, number_of_parliamentarians, number_of_characteristics, number_of_sign):
    while True:
        option_to_enter = input(
            "Виберіть розв'язок: \n 1. Алгоритм лінійної релаксації \n 2. Алгоритм покриття методом 'мінімальний стовпець -максимальний рядок' \n 3. Генетичний алгоритм \n 4. Жадібний алгоритм \n 5. Вихід \n >>>> ")
        if int(option_to_enter) == 1:
            model = LR_alg.LR(matrix_of_parlament)
            print("Розв`язок: {} ".format(model.Solve()))
            print("Значення ЦФ: ", len(model.Solve()))
        elif int(option_to_enter) == 2:
            min_covering_set = mm.find_min_covering_set(matrix_of_parlament)
            print("Розв`язок: {} ".format(min_covering_set))
            print("Значення ЦФ: ", len(min_covering_set))
        elif int(option_to_enter) == 3:
            result = ga.start(matrix_of_parlament, number_of_parliamentarians, number_of_characteristics, number_of_sign)
            print("Розв`язок: {} ".format(result))
            print("Значення ЦФ: ", len(result))
        elif int(option_to_enter) == 4:
            model = Greedy_alg.Greedy(matrix_of_parlament)
            print("Розв`язок: {} ".format(model.Solve()))
            print("Значення ЦФ: ", len(model.Solve()))
        elif int(option_to_enter) == 5:
            break
        else:
            print("Wrong input")


def choose_type_of_enter():
    while True:
        option_to_enter = input(
            "Виберіть опцію: \n 1. Ввести дані вручну \n 2. Згенерувати випадковим чином \n 3. Зчитати з файлу \n 4. Вихід \n >>>> ")
        if int(option_to_enter) == 1:
            matrix_of_parlament, number_of_parliamentarians, number_of_characteristics, number_of_sign = manually_enter()
            choose_algorithm(matrix_of_parlament, number_of_parliamentarians, number_of_characteristics, number_of_sign)
        elif int(option_to_enter) == 2:
            matrix_of_parlament, number_of_parliamentarians, number_of_characteristics, number_of_sign = random_enter()
            choose_algorithm(matrix_of_parlament, number_of_parliamentarians, number_of_characteristics, number_of_sign)
        elif int(option_to_enter) == 3:
            matrix_of_parlament, number_of_parliamentarians, number_of_characteristics, number_of_sign = file_enter()
            choose_algorithm(matrix_of_parlament, number_of_parliamentarians, number_of_characteristics, number_of_sign)
        elif int(option_to_enter) == 4:
            break
        else:
            print("Wrong input")
    return matrix_of_parlament, number_of_parliamentarians, number_of_characteristics, number_of_sign


def choose_type_of_experiment():
    while True:
        option_to_enter = input(
            "Виберіть тип експерименту: \n 1. Алгоритм лінійної релаксації \n 2. Алгоритм покриття методом 'мінімальний стовпець -максимальний рядок' \n 3."
            " Генетичний алгоритм \n 4. Жадібний \n 5. Тест на трудомісткість \n 6. Тест на ефективність \n 7. Вихід \n >>>> ")
        if int(option_to_enter) == 1:
            parl = int(input("Введіть кількість парламентарів: \n"))
            features = int(input("Введіть кількість ознак: \n"))
            exp.LR_exp(parl, features)
        elif int(option_to_enter) == 2:
            print("implement")
        elif int(option_to_enter) == 3:
            matrix_of_parlament, number_of_parliamentarians, number_of_characteristics, number_of_sign = random_enter()
            exp.genetic_alg(matrix_of_parlament)
        elif int(option_to_enter) == 4:
            print("implement")
        elif int(option_to_enter) == 5:
            print("implement")
        elif int(option_to_enter) == 6:
            print("implement")
        elif int(option_to_enter) == 7:
            break
        else:
            print("Wrong input")
if __name__ == '__main__':
    while True:
        option_to_enter = input(
            "Виберіть опцію: \n 1. Робота над індивідуальною задачею  \n 2. Експеримент \n 3. Вихід \n >>>> ")
        if int(option_to_enter) == 1:
            choose_type_of_enter()
        elif int(option_to_enter) == 2:
            choose_type_of_experiment()
        elif int(option_to_enter) == 3:
            break
        else:
            print("Wrong input")
