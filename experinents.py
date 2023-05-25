import numpy as np
import random
import time
import LR_alg
import genetic_algorithm as ga
import matplotlib.pyplot as plt


def copy_matrix(matrix1, matrix2):
    for i in range(len(matrix1)):
        for j in range(len(matrix1[i])):
            matrix2[i][j] = matrix1[i][j]
    return matrix2


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


def LR_exp(parl, features):
    param = [2, round(features / 4) + 1, round(features / 2)]

    time_list = np.zeros((len(param), 20))
    for i in range(0, 20):
        for j in range(0, len(param)):
            matrix = create_matrix_of_parlament(parl, param[j], features)
            start_time = time.time()
            model = LR_alg.LR(matrix)
            model.Solve()
            time
            time_list[j][i] = (time.time() - start_time)

    vals = np.zeros(len(param))
    n = np.arange(0, 20)
    for i in range(0, len(param)):
        vals[i] = np.mean(time_list[i, :])
        plt.plot(n, time_list[i, :], label=param[i])

    print("Середній час виконання для різних знвчень параметру 'кількість характеристик'")
    print(f'{param[0]} : {vals[0]},  {param[1]} : {vals[1]},  {param[2]} : {vals[2]}')

    plt.title("Час виконання алгоритму лінійної релаксації в залежності від кількості характеристик")
    plt.xlabel("Номер прогону")
    plt.ylabel("Час")
    plt.xticks(n)
    plt.grid()
    plt.legend()
    plt.show()


def time_test_n():
    print("Вплив параметру розмірності задачі на трудомісткість алгоритму")
    t = 3
    K_sign = 10
    param = [5, 10, 20]
    print("Задані параметри:")
    print(param)
    print(f"Для {t} характеристик та {K_sign} ознак")
    time_list_ga = np.zeros((len(param), 20))
    time_list_lr = np.zeros((len(param), 20))
    time_min_column_max_row = np.zeros((len(param), 20))
    for i in range(0, 20):
        for j in range(0, len(param)):
            # Генетичний алгоритм
            matrix = create_matrix_of_parlament(param[j], t, K_sign)
            start_time = time.time()
            ga.start(matrix)
            time
            time_list_ga[j][i] = (time.time() - start_time)
            # Алгоритм лінійної релаксації
            start_time = time.time()
            model = LR_alg.LR(matrix)
            model.Solve()
            time_list_lr[j][i] = (time.time() - start_time)
            # Алгоритм мінімальний стовпець -максимальний рядок
            start_time = time.time()
            ga.start(matrix)  # сюда
            time
            time_min_column_max_row[j][i] = (time.time() - start_time)
    vals_ga = np.zeros(len(param))
    for i in range(0, len(param)):
        vals_ga[i] = np.mean(time_list_ga[i, :])
    vals_lr = np.zeros(len(param))
    for i in range(0, len(param)):
        vals_lr[i] = np.mean(time_list_lr[i, :])
    # vals_min_column_max_row = np.zeros(len(param)) #потом це можна прибрати
    # for i in range(0, len(param)):
    #         vals_min_column_max_row[i] = np.mean(time_min_column_max_row[i, :])
    print("Генетичний алгоритм")
    print(vals_ga)
    print("Алгоритм лінійної релаксації")
    print(vals_lr)
    # print("Алгоритм мінімальний стовпець -максимальний рядок") #потом це можна прибрати
    # print(vals_min_column_max_row)


def time_test_k():
    print("Вплив параметру кількості ознак задачі на трудомісткість алгоритму")
    t = 3
    n = 20
    param = [5, 10, 20]
    print("Задані параметри:")
    print(param)
    print(f"Для {t} характеристик та {n} парламентарів")
    time_list_ga = np.zeros((len(param), 20))
    time_list_lr = np.zeros((len(param), 20))
    time_min_column_max_row = np.zeros((len(param), 20))
    for i in range(0, 20):
        for j in range(0, len(param)):
            # Генетичний алгоритм
            matrix = create_matrix_of_parlament(n, t, param[j])
            start_time = time.time()
            ga.start(matrix)
            time
            time_list_ga[j][i] = (time.time() - start_time)
            # Алгоритм лінійної релаксації
            start_time = time.time()
            model = LR_alg.LR(matrix)
            model.Solve()
            time_list_lr[j][i] = (time.time() - start_time)
            # Алгоритм мінімальний стовпець -максимальний рядок
            start_time = time.time()
            ga.start(matrix)  # сюда
            time
            time_min_column_max_row[j][i] = (time.time() - start_time)
    vals_ga = np.zeros(len(param))
    for i in range(0, len(param)):
        vals_ga[i] = np.mean(time_list_ga[i, :])
    vals_lr = np.zeros(len(param))
    for i in range(0, len(param)):
        vals_lr[i] = np.mean(time_list_lr[i, :])
    # vals_min_column_max_row = np.zeros(len(param)) #потом це можна прибрати
    # for i in range(0, len(param)):
    #         vals_min_column_max_row[i] = np.mean(time_min_column_max_row[i, :])
    print("Генетичний алгоритм")
    print(vals_ga)
    print("Алгоритм лінійної релаксації")
    print(vals_lr)
    # print("Алгоритм мінімальний стовпець -максимальний рядок")
    # print(vals_min_column_max_row)


def genetic_alg():
    print("Genetic algorithm experience")
    print("Параметр: умова завершення роботи алгоритмів")
    t = 3
    K_sign = 10
    n = 20
    param = [5, 10, 30]
    matrix = create_matrix_of_parlament(n, t, K_sign)
    population = ga.create_population(matrix)
    population_temp = [[0] * len(population[len(population) - 1]) for _ in range(len(population))]
    population_temp = copy_matrix(population, population_temp)
    iteration_list = np.zeros((len(param), 100))
    value_list = np.zeros((len(param), 100))
    for i in range(100):
        for j in range(0, len(param)):
            population = copy_matrix(population_temp, population)
            iteration_list[j][i], value_list[i] = one_experiment_genetic(matrix, population, param[j])
    ga.print_matrix(iteration_list)
    print("v")
    ga.print_matrix(value_list)
    vals = np.zeros(len(param))
    for i in range(0, len(param)):
        vals[i] = np.mean(iteration_list[i, :])
    print("new")
    print(vals)
    plt.plot(param, vals, label=r'$p = p1$')
    plt.title('$Iteration of stop$')
    plt.legend(loc='upper right', fontsize=6)
    plt.grid(True)
    plt.show()


def one_experiment_genetic(matrix_of_parlament, population, param):
    stop_number_of_iteration = param
    list_of_value1 = []
    list_of_iteration1 = []
    iteration = 0
    number_of_iteration_to_stop = 0
    prev_min_useful = np.min(ga.counter_of_useful(population))
    while True:
        first_parent, second_parent = ga.select_parents(population)
        first_child, second_child = ga.crossingover(matrix_of_parlament, first_parent, second_parent)
        first_child, second_child = ga.mutation(matrix_of_parlament, first_child, second_child)
        new_population = ga.update_population(population, first_child, second_child)
        population = new_population
        min_useful = np.min(ga.counter_of_useful(population))
        if (min_useful == prev_min_useful):
            number_of_iteration_to_stop = number_of_iteration_to_stop + 1
        else:
            prev_min_useful = min_useful
        if (number_of_iteration_to_stop == stop_number_of_iteration):
            break
        iteration = iteration + 1
        list_of_value1.append(np.min(ga.counter_of_useful(population)))
        list_of_iteration1.append(iteration)
    return np.min(ga.counter_of_useful(population)), list_of_value1
