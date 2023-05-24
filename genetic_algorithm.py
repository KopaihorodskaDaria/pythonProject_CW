import numpy as np
import random
import math


def print_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print(matrix[i][j], end='')
        print()

def create_population(matrix):
    rows = math.ceil(len(matrix) / 2)
    columns = len(matrix)
    new_matrix = [[0] * columns for _ in range(rows)]
    for i in range(len(new_matrix)):
        new_matrix[i] = generate_row(new_matrix[i], columns, len(matrix))
    for i in range(len(new_matrix)):
        temp_array = check_condition_of_individ(matrix, new_matrix[i])
        flag = False
        while flag == False:
            for index in range(len(new_matrix)):
                if temp_array == new_matrix[index]:
                    new_matrix[i] = generate_row(new_matrix[i], columns, 2)
                    temp_array = check_condition_of_individ(matrix, new_matrix[i])
                    flag = True
                else:
                    flag = True
        new_matrix[i] = temp_array
    return new_matrix


def generate_row(row, columns, rows):
    amount = random.randint(0, rows - 1)
    ones_indices = random.sample(range(columns), amount)
    for index in ones_indices:
        row[index] = 1
    return row


def counter_of_useful(population):
    array_of_useful = np.zeros(len(population))
    for i in range(len(population)):
        array_of_useful[i] = np.sum(population[i])
    print("array")
    print(array_of_useful)
    return array_of_useful
def counter_of_useful_of_individ(individ):
    useful = np.sum(individ)
    return useful
def get_covered_sign(parlamenters, individ):
    array_of_sign = np.zeros(len(parlamenters[len(parlamenters) - 1]))
    position = 0;
    sign_of_parlamenters_in_commission = np.zeros((len(parlamenters), len(parlamenters[len(parlamenters) - 1])),
                                                  dtype=int)
    for i in range(len(parlamenters)):
        for j in range(len(parlamenters[i])):
            if individ[i] == 1:
                sign_of_parlamenters_in_commission[position] = parlamenters[i]
                position = position + 1
                break
    for i in range(len(sign_of_parlamenters_in_commission)):
        for j in range(len(sign_of_parlamenters_in_commission[i])):
            if individ[i] == 1:
                if np.sum(sign_of_parlamenters_in_commission[:, j]) > 0:
                    array_of_sign[j] = 1
    return array_of_sign

def recreate_the_worst_individ(parlamenters, individ):
    temp_array = individ
    iteration = 0
    while True:
       index = random.randint(0, len(individ) - 1)
       temp_array[index] = 0
       covering_sign = get_covered_sign(parlamenters, temp_array)
       if np.sum(covering_sign) == len(parlamenters[len(parlamenters) - 1]):
           individ = temp_array
           break
       else:
           iteration = iteration + 1
       if iteration == len(individ) - 1:
           break
    return individ

def check_condition_of_individ(parlamenters, individ):
    covering_sign = get_covered_sign(parlamenters, individ)
    while True:
        if np.sum(covering_sign) == len(parlamenters[len(parlamenters) - 1]):
            break
        else:
            index_of_zero = np.where(covering_sign == 0)[0][0]
            for i in range(len(parlamenters)):
                if parlamenters[i][index_of_zero] == 1:
                    individ[i] = 1
            covering_sign = get_covered_sign(parlamenters, individ)
    if np.sum(individ) == len(individ):
        individ = recreate_the_worst_individ(parlamenters, individ)
    return individ

def select_parents(population):
    first_parent_index = random.randint(0, len(population) - 1)
    while True:
        second_parent_index = random.randint(0, len(population) - 1)
        if (second_parent_index != first_parent_index):
            break
    first_parent = population[first_parent_index]
    second_parent = population[second_parent_index]
    return first_parent, second_parent


def crossingover(parlamenters, first_parent, second_parent):
    crossing_point = random.randint(0, len(parlamenters) - 1)
    temp1 = first_parent[:crossing_point]
    temp2 = second_parent[crossing_point:]
    print("A random point of one point crossingover: ", crossing_point)
    child_1 = np.zeros(len(parlamenters), dtype=int)
    child_2 = np.zeros(len(parlamenters), dtype=int)
    new_index1 = len(temp1)
    new_index2 = len(temp2)
    for i in range(len(temp1)):
        child_1[i] = temp1[i]
    for i in range(len(temp2)):
        child_1[new_index1] = temp2[i]
        new_index1 = new_index1 + 1
    for i in range(len(temp2)):
        child_2[i] = temp2[i]
    for i in range(len(temp1)):
        child_2[new_index2] = temp1[i]
        new_index2 = new_index2 + 1
    child_1 = check_condition_of_individ(parlamenters, child_1)
    child_2 = check_condition_of_individ(parlamenters, child_2)
    return child_1, child_2


def mutation(parlamenters, first_child, second_child):
    num1 = random.randint(1, 2)
    num2 = random.randint(1, 2)
    if num1 == 2:
        print("First child mutated")
        mutating_point = random.randint(0, len(first_child) - 1)
        for i in range(len(first_child)):
            if mutating_point == first_child[i]:
                if first_child[i] == 1:
                    first_child[i] = 0
                else:
                    first_child[i] = 1
        first_child = check_condition_of_individ(parlamenters, first_child)
        print(first_child)
    elif num2 == 2:
        print("Second child mutated")
        mutating_point = random.randint(0, len(second_child) - 1)
        for i in range(len(second_child)):
            if mutating_point == second_child[i]:
                if second_child[i] == 1:
                    second_child[i] = 0
                else:
                    second_child[i] = 1
        second_child = check_condition_of_individ(parlamenters, second_child)
        print(second_child)
    else:
        print("No one mutated")
    return first_child, second_child


def update_population(population, first_child, second_child):
    useful_per_person = counter_of_useful(population)
    if counter_of_useful_of_individ(first_child) < np.max(useful_per_person):
        first_index_of_individ = np.argmax(useful_per_person)
        for j in range(len(population[len(population)-1])):
                population[first_index_of_individ][j] = first_child[j]
        useful_per_person = counter_of_useful(population)
    if counter_of_useful_of_individ(second_child) < np.max(useful_per_person):
        second_index_of_individ = np.argmax(useful_per_person)
        for j in range(len(population[len(population) -1])):
            population[second_index_of_individ][j] = second_child[j]
    return population

def get_one_solution(matrix_of_parlament, population ):
    iteration = 0
    number_of_iteration_to_stop = 0
    stop_number_of_iteration = 20
    prev_min_useful = np.min(counter_of_useful(population))
    print("The value of the objective function: ", prev_min_useful)
    while True:
        print(f'----- Iteration № {iteration} -----')

        print("Selected parents")
        first_parent, second_parent = select_parents(population)
        print("First parent: ", first_parent)
        print("Second parent: ", second_parent)

        print("Crossingover")
        first_child, second_child = crossingover(matrix_of_parlament, first_parent, second_parent)
        print("First child: ", first_child)
        print("Second child: ", second_child)

        print("Mutation with 50% probability")
        first_child, second_child = mutation(matrix_of_parlament, first_child, second_child)

        print("Update population")
        new_population = update_population(population, first_child, second_child)
        population = new_population
        print_matrix(population)

        min_useful = np.min(counter_of_useful(population))
        print("The value of the objective function: ", min_useful)
        print("Individ with the best value of the objective function: ", population[np.argmin(counter_of_useful(population))])
        if(min_useful == prev_min_useful):
            number_of_iteration_to_stop = number_of_iteration_to_stop + 1
        else:
            prev_min_useful = min_useful
        if(number_of_iteration_to_stop == stop_number_of_iteration):
            break
        iteration = iteration + 1

def start(matrix_of_parlament, number_of_parliamentarians, number_of_characteristics, number_of_sign):
      print("Genetic algorithm")
      population = create_population(matrix_of_parlament)
      print("Create matrix of individ in population")
      print_matrix(population)
      get_one_solution(matrix_of_parlament, population)