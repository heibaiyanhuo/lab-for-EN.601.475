import math
import numpy as np


class AdaBoostHelper:
    def __init__(self, X, y, n, m):
        self.cache = dict()
        self.X = X.toarray()

        self.y = np.ones(n)
        self.y[y == 0] = -1
        self.n = n
        self.m = m
        self.D = np.array([1 / n for i in range(n)])

    def calc_yhat(self, j, c):
        y_hat_arr = np.ones(self.n)
        dimension = self.X[:, j]
        y_hat_arr[dimension <= c] = -1

        if np.sum(y_hat_arr == self.y) < self.n / 2:
            return -y_hat_arr, -1
        return y_hat_arr, 1

    def calc_ht(self):
        j_t = None
        c_t = None
        y_hat_arr_t = None
        y_hat_val_t = None
        error = 1

        for j in range(self.m):
            j_arr_sorted = None
            if j in self.cache:
                j_arr_sorted = self.cache[j]
            else:
                # j_arr_sorted = np.sort(np.unique(self.X[:, j]))
                j_arr_sorted = np.sort(self.X[:, j])
                self.cache[j] = j_arr_sorted

            c_list = np.unique((j_arr_sorted[:self.n - 1] + j_arr_sorted[1:]) / 2)
            # print(c_list)
            # jl = len(j_arr_sorted)
            # print(jl)

            for c in c_list:
                # c = (j_arr_sorted[k] + j_arr_sorted[k + 1]) / 2
                y_hat_arr, y_hat_val = self.calc_yhat(j, c)
                curr_error_arr = np.zeros(self.n)
                curr_error_arr[y_hat_arr != self.y] = 1
                curr_error = np.dot(self.D, curr_error_arr)
                # print('j: {}, k: {}, c: {}, e: {}'.format(j, k, c, curr_error))

                if curr_error < error:
                    j_t = j
                    c_t = c
                    y_hat_arr_t = y_hat_arr
                    y_hat_val_t = y_hat_val
                    error = curr_error

        return j_t, c_t, y_hat_arr_t, y_hat_val_t, error

    def calc_alphat(self, error):
        return 0.5 * math.log((1 / error) - 1)

    def set_next_distribution(self, alpha, y_hat):
        self.D = np.multiply(self.D, np.exp(np.multiply(-alpha * self.y, y_hat)))
        Z = self.D.sum()
        self.D /= Z

    @staticmethod
    def ht(X, h_arg):
        num_of_features = X.shape[0]
        j, c, y_hat_arr, y_hat_val, error = h_arg
        res = np.zeros(num_of_features)
        dimension = X[:, j]
        res[dimension > c] = y_hat_val
        res[dimension <= c] = -y_hat_val
        return res
