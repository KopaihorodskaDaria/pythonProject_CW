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

def LR_exp():
    param = [2, 3, 5]

    time_list = np.zeros((len(param), 20))
    for i in range(0, 20):
        for j in range(0, len(param)):
            matrix = create_matrix_of_parlament(10, param[j], 10)
            start_time = time.time()
            model = LR_alg.LR(matrix)
            model.Solve()
            time
            time_list[j][i] = (time.time()-start_time)

    vals = np.zeros(len(param))
    for i in range(0, len(param)):
        vals[i]= np.mean(time_list[i, :])

    print(vals)
# def time_test():

def genetic_alg(matrix_of_parlament):
    print("Genetic algorithm experience")
    print("Параметр: умова завершення роботи алгоритмів")
    population = ga.create_population(matrix_of_parlament)
    for i in range(5):
     one_experiment_genetic_alg(matrix_of_parlament, population)

def one_experiment_genetic_alg(matrix_of_parlament, population):
    p1 = 3
    p2 = 8
    p3 = 15
    population_temp = [[0] * len(population[len(population) - 1]) for _ in range(len(population))]
    population_temp = copy_matrix(population, population_temp)
    stop_number_of_iteration = [p1, p2, p3]
    list_of_value1 = []
    list_of_iteration1 = []
    list_of_value2 = []
    list_of_iteration2 = []
    list_of_value3 = []
    list_of_iteration3 = []
    for p in range(len(stop_number_of_iteration)):
       iteration = 0
       number_of_iteration_to_stop = 0
       population = copy_matrix(population_temp, population)
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
        if (number_of_iteration_to_stop == stop_number_of_iteration[p]):
            break
        iteration = iteration + 1
        if p == 0:
          list_of_value1.append(np.min(ga.counter_of_useful(population)))
          list_of_iteration1.append(iteration)
        if p == 1:
          list_of_value2.append(np.min(ga.counter_of_useful(population)))
          list_of_iteration2.append(iteration)
        if p == 2:
          list_of_value3.append(np.min(ga.counter_of_useful(population)))
          list_of_iteration3.append(iteration)

    plt.plot(list_of_iteration1, list_of_value1, label = r'$p = p1$')
    plt.plot(list_of_iteration2, list_of_value2,  label = r'$p = p2$')
    plt.plot(list_of_iteration3, list_of_value3,  label = r'$p = p3$')
    plt.title('$Iteration of stop$')
    plt.legend(loc = 'upper right', fontsize = 6)
    plt.grid(True)
    plt.show()




