import numpy as np
import random
import time
import LR_alg


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





