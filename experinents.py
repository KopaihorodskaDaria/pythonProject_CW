import statistics

import numpy as np
import random
import time

import Greedy_alg
import LR_alg
import genetic_algorithm as ga
import matplotlib.pyplot as plt

import min_column_max_row


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
            model = LR_alg.LR(matrix)
            model.Solve()
            time_list[j][i] = model.n

    vals = np.zeros(len(param))
    n = np.arange(0, 20)
    for i in range(0, len(param)):
        vals[i] = np.mean(time_list[i, :])
        plt.plot(n, time_list[i, :], label=param[i])

    print("Середня кількість ітерацій для різних значень параметру 'кількість характеристик'")
    print(f'{param[0]} : {vals[0]},  {param[1]} : {vals[1]},  {param[2]} : {vals[2]}')

    plt.title("Кількість ітерацій алгоритму лінійної релаксації в залежності від кількості характеристик")
    plt.xlabel("Номер прогону")
    plt.ylabel("Кількість ітерацій")
    plt.xticks(n)
    plt.grid()
    plt.legend()
    plt.show()
    parameters = [str(x) for x in param]
    plt.title("Середня кількість ітерацій алгоритму в залежності від кількості характеристик")
    plt.ylabel("Кількість ітерацій")
    plt.bar(parameters, vals)
    plt.grid()
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
            time_list_ga[j][i] = (time.time() - start_time)
            # Алгоритм лінійної релаксації
            start_time = time.time()
            model = LR_alg.LR(matrix)
            model.Solve()
            time_list_lr[j][i] = (time.time() - start_time)
            # Алгоритм мінімальний стовпець -максимальний рядок
            start_time = time.time()
            min_column_max_row.find_min_covering_set(matrix)  # сюда
            time
            time_min_column_max_row[j][i] = (time.time() - start_time)
    vals_ga = np.zeros(len(param))
    for i in range(0, len(param)):
        vals_ga[i] = np.mean(time_list_ga[i, :])
    vals_lr = np.zeros(len(param))
    for i in range(0, len(param)):
        vals_lr[i] = np.mean(time_list_lr[i, :])
    vals_min_column_max_row = np.zeros(len(param))  # потом це можна прибрати
    for i in range(0, len(param)):
        vals_min_column_max_row[i] = np.mean(time_min_column_max_row[i, :])
    print("Генетичний алгоритм")
    print(vals_ga)
    print("Алгоритм лінійної релаксації")
    print(vals_lr)
    print("Алгоритм мінімальний стовпець -максимальний рядок")  # потом це можна прибрати
    print(vals_min_column_max_row)


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
            min_column_max_row.find_min_covering_set(matrix)  # сюда
            time
            time_min_column_max_row[j][i] = (time.time() - start_time)
    vals_ga = np.zeros(len(param))
    for i in range(0, len(param)):
        vals_ga[i] = np.mean(time_list_ga[i, :])
    vals_lr = np.zeros(len(param))
    for i in range(0, len(param)):
        vals_lr[i] = np.mean(time_list_lr[i, :])
    vals_min_column_max_row = np.zeros(len(param))  # потом це можна прибрати
    for i in range(0, len(param)):
        vals_min_column_max_row[i] = np.mean(time_min_column_max_row[i, :])
    print("Генетичний алгоритм")
    print(vals_ga)
    print("Алгоритм лінійної релаксації")
    print(vals_lr)
    print("Алгоритм мінімальний стовпець -максимальний рядок")
    print(vals_min_column_max_row)


def genetic_alg(n , t, K_sign, param ):
    print("Генетичний експеримент")
    print("Параметр: умова завершення роботи алгоритмів")
    iteration = 20
    # population_temp = [[0] * len(population[len(population) - 1]) for _ in range(len(population))]
    # population_temp = copy_matrix(population, population_temp)
    iteration_list = np.zeros((len(param), iteration))
    for i in range(iteration):
        matrix = create_matrix_of_parlament(n, t, K_sign)
        population = ga.create_population(matrix)
        for j in range(0, len(param)):
            iteration_list[j][i] = one_experiment_genetic(matrix, population, param[j])
    ga.print_matrix(iteration_list)
    m = np.arange(0, iteration)
    vals = np.zeros(len(param))
    for i in range(0, len(param)):
        vals[i] = np.mean(iteration_list[i, :])
        plt.plot(m, iteration_list[i, :], label=param[i])
    plt.title("Залежність значення ЦФ від кількості ітерацій")
    plt.legend(loc='upper right', fontsize=6)
    plt.xlabel("Кількість ітерацій")
    plt.ylabel("Значення ЦФ")
    plt.grid(True)
    plt.show()
    print("Середнє значення ЦФ для різних значень параметру 'кількість ітерацій до зупинки'")
    for i in range(len(param)):
        print(f'{param[i]} : {vals[i]}')
    parameters = [str(x) for x in param]
    plt.title("Середнє значення ЦФ відповідно до введеного параметра")
    plt.xlabel("Кількість ітерацій")
    plt.ylabel("Значення ЦФ")
    plt.bar(parameters, vals)
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
    return np.min(ga.counter_of_useful(population))


def min_c_max_r_experiment():
    n_values = [5, 15, 35]  # Кількість рядків (парламентарів)
    K = 10  # Кількість стовпців
    t = 2
    runs = 20  # Кількість прогонів для кожного значення парламентарів

    results = []  # Збереження результатів

    for n in n_values:
        total_times = []
        print(f"Експеримент для кількості парламентарів n = {n}:")
        for run in range(1, runs + 1):
            matrix = create_matrix_of_parlament(n, t, K)
            start_time = time.time()
            min_column_max_row.find_min_covering_set(matrix)
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_times.append(elapsed_time)
            print(f"Прогін {run}: {elapsed_time:.8f} ")

        avg_time = sum(total_times) / runs
        print(f"Середній час виконання для n = {n}: {avg_time} ")
        results.append(total_times)

    return n_values, results


def plot_results(n_values, results):
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', 'D']
    for i in range(len(n_values)):
        plt.plot(range(1, 21), results[i], marker=markers[i], label=f'n = {n_values[i]}')
    plt.xticks(range(1, 21))
    plt.xlabel('Номер прогону')
    plt.ylabel('Час виконання (секунди)')
    plt.title('Час виконання алгоритму в залежності від "кількості парламентарів"( дослідження трудомісткості алгоритму)')
    plt.legend()
    plt.grid(True)
    plt.show()

def min_max_experiment2():
    n_values = []
    n_count = int(input("Введіть кількість значень для масиву n_values: "))
    for i in range(n_count):
        n = int(input(f"Введіть значення для елементу n_values[{i}]: "))
        n_values.append(n)  # Кількість рядків (парламентарів)
    K = int(input("Введіть кількість стовпців K: "))  # Кількість стовпців
    t = int(input("Введіть значення t: "))
    runs = 20  # Кількість прогонів для кожного значення парламентарів
    results = []  # Збереження результатів
    cf_values = []  # Збереження значень CF

    for run in range(1, runs + 1):
        total_times = []
        cf_values_n = []  # Значення CF для даного n
        print(f"Прогін {run}:")
        for n in n_values:
            matrix = create_matrix_of_parlament(n, t, K)
            start_time = time.time()
            covering_set = min_column_max_row.find_min_covering_set(matrix)
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_times.append(elapsed_time)
            cf = len(covering_set)
            cf_values_n.append(cf)
            print(f"Експеримент для кількості парламентарів n = {n}: {elapsed_time:.8f}, ЦФ = {cf}")

        avg_time = sum(total_times) / len(n_values)
        print(f"Середній час виконання: {avg_time}")
        results.append(total_times)
        cf_values.append(cf_values_n)
        avg_cf = sum(cf_values_n) / len(cf_values_n)
        print(f"Середнє значення ЦФ: {avg_cf}")
        std_cf = statistics.stdev(cf_values_n)
        print(f"Стандартне відхилення значень ЦФ: {std_cf}")

    # Plotting CF values against the number of runs
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', 'D']
    for i in range(len(n_values)):
        plt.plot(range(1, runs + 1), [cf_values[j][i] for j in range(runs)], marker=markers[i], label=f'n = {n_values[i]}')
    plt.xticks(range(1, runs + 1))
    plt.xlabel('Прогін')
    plt.ylabel('Значення ЦФ')
    plt.title('Залежність значення ЦФ від прогону (дослідження ефективності алгоритму)')
    plt.legend()
    plt.grid(True)
    plt.show()

def precision_test_1():
    global precision_list_lr, precision_list_min_column_max_row
    t = 3
    K_sign = 6
    param = [5, 10, 15]
    print("Вплив параметру розмірності задачі на ефективність алгоритму")
    print("Задані параметри:")
    print(param)
    print(f"Для {t} характеристик та {K_sign} ознак")

    deviations_ga = []
    deviations_lr = []
    deviations_min_column_max_row = []
    deviations_gr = []

    for i in range(1, 21):
        avg_ga = 0
        avg_lr = 0
        avg_min_column_max_row = 0
        avg_gr = 0

        for j in range(len(param)):
            matrix = create_matrix_of_parlament(param[j], t, K_sign)

            # Генетичний алгоритм
            result_ga = ga.start(matrix)
            precision_list_ga = len(result_ga)
            avg_ga += precision_list_ga

            # Алгоритм лінійної релаксації
            model = LR_alg.LR(matrix)
            result_lr = model.Solve()
            precision_list_lr = len(result_lr)
            avg_lr += precision_list_lr

            # Алгоритм мінімальний стовпець - максимальний рядок
            result_min_column_max_row = min_column_max_row.find_min_covering_set(matrix)
            precision_list_min_column_max_row = len(result_min_column_max_row)
            avg_min_column_max_row += precision_list_min_column_max_row

            # Жадібний алгоритм
            model = Greedy_alg.Greedy(matrix)
            result_gr = model.Solve()
            precision_list_gr = len(result_gr)
            avg_gr += precision_list_gr
            avg_ga /= 20
            avg_lr /= 20
            avg_min_column_max_row /= 20
            avg_gr /= 20
            deviations_ga.append(avg_ga)
            deviations_lr.append(avg_lr)
            deviations_min_column_max_row.append(avg_min_column_max_row)
            deviations_gr.append(avg_gr)

    deviation_ga_avg = np.mean(deviations_ga)
    deviation_lr_avg = np.mean(deviations_lr)
    deviation_min_column_max_row_avg = np.mean(deviations_min_column_max_row)
    deviation_gr_avg = np.mean(deviations_gr)

    print("Average Deviations:")
    print("Genetic Algorithm:", deviation_ga_avg)
    print("Linear Relaxation Algorithm:", deviation_lr_avg)
    print("Minimum Column Maximum Row Algorithm:", deviation_min_column_max_row_avg)
    print("Greedy Algorithm:", deviation_gr_avg)


def precision_test_2():
    global precision_list_lr, precision_list_min_column_max_row
    t = 3
    n = 20
    param = [6, 12, 18]
    print("Вплив параметру розмірності задачі на ефективність алгоритму")
    print("Задані параметри:")
    print(param)
    print(f"Для {t} характеристик та {n} ознак")

    deviations_ga = []
    deviations_lr = []
    deviations_min_column_max_row = []
    deviations_gr = []

    for i in range(1, 21):
        avg_ga = 0
        avg_lr = 0
        avg_min_column_max_row = 0
        avg_gr = 0

        for j in range(len(param)):
            matrix = create_matrix_of_parlament(n, t, param[j])

            # Генетичний алгоритм
            result_ga = ga.start(matrix)
            precision_list_ga = len(result_ga)
            avg_ga += precision_list_ga

            # Алгоритм лінійної релаксації
            model = LR_alg.LR(matrix)
            result_lr = model.Solve()
            precision_list_lr = len(result_lr)
            avg_lr += precision_list_lr

            # Алгоритм мінімальний стовпець - максимальний рядок
            result_min_column_max_row = min_column_max_row.find_min_covering_set(matrix)
            precision_list_min_column_max_row = len(result_min_column_max_row)
            avg_min_column_max_row += precision_list_min_column_max_row

            # Жадібний алгоритм
            model = Greedy_alg.Greedy(matrix)
            result_gr = model.Solve()
            precision_list_gr = len(result_gr)
            avg_gr += precision_list_gr

            avg_ga /= 20
            avg_lr /= 20
            avg_min_column_max_row /= 20
            avg_gr /= 20
            deviations_ga.append(avg_ga)
            deviations_lr.append(avg_lr)
            deviations_min_column_max_row.append(avg_min_column_max_row)
            deviations_gr.append(avg_gr)

    deviation_ga_avg = np.mean(deviations_ga)
    deviation_lr_avg = np.mean(deviations_lr)
    deviation_min_column_max_row_avg = np.mean(deviations_min_column_max_row)
    deviation_gr_avg = np.mean(deviations_gr)

    print("Average Deviations:")
    print("Genetic Algorithm:", deviation_ga_avg)
    print("Linear Relaxation Algorithm:", deviation_lr_avg)
    print("Minimum Column Maximum Row Algorithm:", deviation_min_column_max_row_avg)
    print("Greedy Algorithm:", deviation_gr_avg)


def precision_test_3():
    global CFsum_min_column_max_row, CFsum_lr, CFsum_ga, CFsum_gr
    print("Дослідження впливу кількості парламентарів на ефективність алгоритмів")
    t = 10  # Розмірність задачі
    K = 20  # кількість ознак
    param = [10, 15, 20]
    list_cf = np.ones((3, 20))

    for j in range(len(param)):
        CFsum_lr = 0
        CFsum_min_column_max_row = 0
        CFsum_ga = 0
        CFavg = 0
        CFsum_gr = 0

        for i in range(1, 21):
            matrix = create_matrix_of_parlament(param[j], t,K )
            # Розв'язок задачі P алгоритмом "Жадібний алгоритм"
            model = Greedy_alg.Greedy(matrix)
            CF_gr = model.Solve()
            CFsum_gr += len(CF_gr)

            # Розв'язок задачі P алгоритмом "Лінійної релаксації"
            model = LR_alg.LR(matrix)
            CF_lr = model.Solve()
            CFsum_lr += len(CF_lr)

            # Розв'язок задачі P алгоритмом "Мінімальний стовпець - максимальний рядок"
            CF_min_column_max_row = min_column_max_row.find_min_covering_set(matrix)
            CFsum_min_column_max_row += len(CF_min_column_max_row)

            # Розв'язок задачі P алгоритмом "Генетичний алгоритм"
            CF_ga = ga.start(matrix)
            CFsum_ga += len(CF_ga)

            # Розв'язок задачі P алгоритмом "Жадібний алгоритм"
            model = Greedy_alg.Greedy(matrix)
            CF_gr = model.Solve()
            CFsum_gr += len(CF_gr)

        CFavg1 = CFsum_lr / 20
        CFavg2 = CFsum_ga / 20
        CFavg3 = CFsum_min_column_max_row / 20
        CFavg4 = CFsum_gr / 20
        CFAvr = (CFavg1+ CFavg2 +  CFavg3+  CFavg4)/4
        CF1 = abs(CFavg1 - CFAvr) / CFAvr
        CF2 = abs(CFavg2 - CFAvr) / CFAvr
        CF3 = abs(CFavg3 - CFAvr) / CFAvr
        CF4 = abs(CFavg4 - CFAvr) / CFAvr

        print(f"Параметр = {param[j]}:")
        print(f"Лінійна релаксація: {CF1}")
        print(f"Генетичний алгоритм: {CF2}")
        print(f"Мінімальний стовпець - максимальний рядок: {CF3}")
        print(f"Жадібний алгоритм: {CF4}")
        print()



def precision_test_4():
    global CFsum_min_column_max_row, CFsum_lr, CFsum_ga, CFsum_gr
    print("Дослідження впливу кількості ознак на ефективність алгоритмів")
    t = 5
    n = 20
    param = [10, 15, 20]
    list_cf = np.ones((3, 20))

    for j in range(len(param)):
        CFsum_lr = 0
        CFsum_min_column_max_row = 0
        CFsum_ga = 0
        CFavg = 0
        CFsum_gr = 0

        for i in range(1, 21):
            matrix = create_matrix_of_parlament(n, t, param[j])
            # Розв'язок задачі P алгоритмом "Жадібний алгоритм"
            model = Greedy_alg.Greedy(matrix)
            CF_gr = model.Solve()
            CFsum_gr += len(CF_gr)

            # Розв'язок задачі P алгоритмом "Лінійної релаксації"
            model = LR_alg.LR(matrix)
            CF_lr = model.Solve()
            CFsum_lr += len(CF_lr)

            # Розв'язок задачі P алгоритмом "Мінімальний стовпець - максимальний рядок"
            CF_min_column_max_row = min_column_max_row.find_min_covering_set(matrix)
            CFsum_min_column_max_row += len(CF_min_column_max_row)

            # Розв'язок задачі P алгоритмом "Генетичний алгоритм"
            CF_ga = ga.start(matrix)
            CFsum_ga += len(CF_ga)

            # Розв'язок задачі P алгоритмом "Жадібний алгоритм"
            model = Greedy_alg.Greedy(matrix)
            CF_gr = model.Solve()
            CFsum_gr += len(CF_gr)

        CFavg1 = CFsum_lr / 20
        CFavg2 = CFsum_ga / 20
        CFavg3 = CFsum_min_column_max_row / 20
        CFavg4 = CFsum_gr / 20
        CFAvr = (CFavg1+ CFavg2 +  CFavg3+  CFavg4)/4
        CF1 = abs(CFavg1 - CFAvr) / CFAvr
        CF2 = abs(CFavg2 - CFAvr) / CFAvr
        CF3 = abs(CFavg3 - CFAvr) / CFAvr
        CF4 = abs(CFavg4 - CFAvr) / CFAvr

        print(f"Параметр = {param[j]}:")
        print(f"Лінійна релаксація: {CF1}")
        print(f"Генетичний алгоритм: {CF2}")
        print(f"Мінімальний стовпець - максимальний рядок: {CF3}")
        print(f"Жадібний алгоритм: {CF4}")
        print()


def precision_test_5():
        global CFsum_min_column_max_row, CFsum_lr, CFsum_ga, CFsum_gr
        print("Дослідження впливу параметрів задачі на ефективність алгоритмів")
        n = 30
        K = 30
        param = [2, 7, 15]
        list_cf = np.ones((3, 20))

        for j in range(len(param)):
            CFsum_lr = 0
            CFsum_min_column_max_row = 0
            CFsum_ga = 0
            CFavg = 0
            CFsum_gr = 0

            for i in range(1, 21):
                matrix = create_matrix_of_parlament(n, param[j], K)
                # Розв'язок задачі P алгоритмом "Жадібний алгоритм"
                model = Greedy_alg.Greedy(matrix)
                CF_gr = model.Solve()
                CFsum_gr += len(CF_gr)

                # Розв'язок задачі P алгоритмом "Лінійної релаксації"
                model = LR_alg.LR(matrix)
                CF_lr = model.Solve()
                CFsum_lr += len(CF_lr)

                # Розв'язок задачі P алгоритмом "Мінімальний стовпець - максимальний рядок"
                CF_min_column_max_row = min_column_max_row.find_min_covering_set(matrix)
                CFsum_min_column_max_row += len(CF_min_column_max_row)

                # Розв'язок задачі P алгоритмом "Генетичний алгоритм"
                CF_ga = ga.start(matrix)
                CFsum_ga += len(CF_ga)

                # Розв'язок задачі P алгоритмом "Жадібний алгоритм"
                model = Greedy_alg.Greedy(matrix)
                CF_gr = model.Solve()
                CFsum_gr += len(CF_gr)

            CFavg1 = CFsum_lr / 20
            CFavg2 = CFsum_ga / 20
            CFavg3 = CFsum_min_column_max_row / 20
            CFavg4 = CFsum_gr / 20
            CFAvr = (CFavg1 + CFavg2 + CFavg3 + CFavg4) / 4
            CF1 = abs(CFavg1 - CFAvr) / CFAvr
            CF2 = abs(CFavg2 - CFAvr) / CFAvr
            CF3 = abs(CFavg3 - CFAvr) / CFAvr
            CF4 = abs(CFavg4 - CFAvr) / CFAvr

            print(f"Параметр = {param[j]}:")
            print(f"Лінійна релаксація: {CF1}")
            print(f"Генетичний алгоритм: {CF2}")
            print(f"Мінімальний стовпець - максимальний рядок: {CF3}")
            print(f"Жадібний алгоритм: {CF4}")
            print()

    # # Побудова графіка
    # x = np.arange(1, 21)
    #
    # plt.plot(x, precision_list_ga[0], label="Genetic Algorithm")
    # plt.plot(x, precision_list_lr[0], label="Linear Relaxation Algorithm")
    # plt.plot(x, precision_min_column_max_row[0], label="Min Column - Max Row Algorithm")
    # plt.plot(x, precision_list_gr[0], label="Greedy Algorithm")
    #
    # plt.xlabel("Run")
    # plt.ylabel("Precision")
    # plt.title("Precision vs Run")
    # plt.legend()
    # plt.show()
