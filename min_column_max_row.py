def min_column(y, n, K, active_columns):
    min_count = float('inf')
    min_column_index = -1

    for j in range(K):
        if active_columns[j] == 1:
            count = sum(y[i][j] for i in range(n))
            if count < min_count:
                min_count = count
                min_column_index = j

    return min_column_index


def max_row(y, is_min, n, K, active_rows):
    max_count = -1
    max_row_index = -1

    for i in range(n):
        if active_rows[i] == 1 and y[i][is_min] == 1:
            count = sum(y[i][j] for j in range(K))
            if count > max_count:
                max_count = count
                max_row_index = i

    return max_row_index


def note_and_delete(arr_coverage, is_max, y, K, active_columns):
    arr_coverage.append(is_max)

    for j in range(K):
        if active_columns[j] == 1 and y[is_max][j] == 1:
            active_columns[j] = 0


def find_min_covering_set(y):
    n = len(y)  # Кількість парламентарів
    K = len(y[0])  # Кількість ознак

    arr_coverage = []  # Масив покриття
    arr_coverage_seq = []  # Масив покриття
    active_columns = [1] * K  # Активні стовпці
    active_rows = [1] * n  # Активні рядки

    while 1 in active_columns:
        is_min = min_column(y, n, K, active_columns)
        is_max = max_row(y, is_min, n, K, active_rows)
        note_and_delete(arr_coverage, is_max, y, K, active_columns)
        arr_coverage_seq.append(is_max + 1)

    return arr_coverage_seq

