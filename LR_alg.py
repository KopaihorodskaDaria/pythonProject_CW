import numpy as np
import random
import math
import sys

class LR:
    def __init__(self, A):
        self.A = np.array(A)
        self.table, self.basis = self.tableau()

    def adapt(self):
        self.A = self.A.T * -1

        b = np.ones(self.A.shape[0]) * -1
        b = np.concatenate((b, [0]), axis=0)
        b = b.reshape(1, b.shape[0])
        c = np.ones(self.A.shape[1])
        c = c.reshape(1, c.shape[0])

        return b, c

    def tableau(self):
        b, c = self.adapt()
        tableau = np.concatenate((self.A, c), axis=0)
        bas = np.concatenate((np.eye(self.A.shape[0]), [np.zeros(self.A.shape[0])]), axis=0)
        tableau = np.concatenate((tableau, bas), axis=1)
        tableau = np.concatenate((tableau, b.T), axis=1)
        basis = np.arange(self.A.shape[1] + 1, tableau.shape[1])

        return tableau, basis

    def Solve(self):
        b_n = self.table[:, -1]
        while any(b_n[0:-1] < 0):
            self.change_b()
            b_n = self.table[:, -1]
        z_n = self.table[-1, :]
        while any(z_n[0: self.table.shape[0] - 1] < 0):
            l = self.change_z()
            if l != 1:
                break
            z_n = self.table[-1, :]

        result = np.zeros(self.A.shape[1])
        b_n = self.table[:, -1]
        for i in range(0, len(self.basis)):
            if (self.basis[i] in np.arange(1, self.A.shape[1] + 1)):
                result[self.basis[i] - 1] = b_n[i]

        result = self.rounding(result)

        res = np.flatnonzero(result == np.max(result))
        res = res + 1
        return res

    def change_z(self):
        z_n = self.table[-1, :]
        column = z_n[0: self.table.shape[1] - 1].argmin()
        column_list = self.table[:, column]
        if (all(n <= 0 for n in column_list[0:-1])):
            return 0
        b_n = self.table[:, -1]
        column_b = np.array([])
        for i in range(0, len(column_list[0: -1])):
            if (column_list[i] > 0):
                column_b = np.append(column_b, b_n[i] / column_list[i])
            else:
                column_b = np.append(column_b, math.inf)
        if (all(n == math.inf for n in column_b)):
            return 0
        row = column_b.argmin()
        row_list = self.table[row, :] / self.table[row, column]
        for i in range(0, self.table.shape[0]):
            if (i == row):
                self.table[i, :] = row_list
            else:
                self.table[i, :] = self.table[i, :] - row_list * self.table[i, column]
        self.basis[row] = (column + 1)
        return 1

    def change_b(self):
        b_n = self.table[:, -1]
        row = b_n[0:-1].argmin()
        row_list = self.table[row, :]
        column = self.table[row, 0:-1].argmin()
        row_list = row_list / self.table[row, column]

        for i in range(0, self.table.shape[0]):
            if (i == row):
                self.table[i, :] = row_list
            else:
                self.table[i, :] = self.table[i, :] - row_list * self.table[i, column]
        self.basis[row] = (column + 1)

    def rounding(self, result):
        A = self.A * -1
        round_v = []
        power = []
        for i in range(0, A.shape[0]):
            power.append(sum(A[i, :]))

        for i in range(0, A.shape[1]):
            ind = []
            for j in range(A.shape[0]):
                if (A[j][i] != 0):
                    ind.append(j)
            ind_pow = np.take(power, ind)
            f = ind_pow.max()

            round_v.append(1 / f)

        for i in range(0, len(result)):
            if (result[i] >= round_v[i]):
                result[i] = 1
            else:
                result[i] = 0

        return result
