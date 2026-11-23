import numpy as np
from scipy.interpolate import CubicSpline

class CubicSplineInterpolator:
    def __init__(self, x, y, d):
        self._cubic_splines = []
        self.x = x
        for i in range(len(x) - 1):
            x_l, x_r = x[i], x[i + 1]
            y_l, y_r = y[i], y[i + 1]
            d_l, d_r = d[i], d[i + 1]
            self._cubic_splines.append(CubicSpline([x_l, x_r], [y_l, y_r], bc_type=((1, d_l), (1, d_r))))

    def __call__(self, x):
        if isinstance(x, float):
            x = [x]
        output = []
        for x_ in x:
            if self.x[0] > x_:
                output.append(self._cubic_splines[0](x_))
            if self.x[-1] <= x_:
                output.append(self._cubic_splines[-1](x_))
            for i in range(len(self.x) - 1):
                if self.x[i] <= x_ < self.x[i + 1]:
                    output.append(self._cubic_splines[i](x_))
        return np.array(output)

    def derivative(self, x, nu=1):
        if isinstance(x, float):
            x = [x]
        output = []
        for x_ in x:
            if self.x[0] > x_:
                output.append(self._cubic_splines[0].derivative(nu=nu)(x_))
            if self.x[-1] <= x_:
                output.append(self._cubic_splines[-1].derivative(nu=nu)(x_))
            for i in range(len(self.x) - 1):
                if self.x[i] <= x_ < self.x[i + 1]:
                    output.append(self._cubic_splines[i].derivative(nu=nu)(x_))
        return np.array(output)

class RationalCubicSplineInterpolator:
    def __init__(self, x, y, r, d):
        self._rational_cubic_splines = []
        self._segments = []
        self.x = np.asarray(x)

        for i in range(len(x) - 1):
            x_l, x_r = x[i], x[i + 1]
            y_l, y_r = y[i], y[i + 1]
            r_i = r[i]
            d_l, d_r = d[i], d[i + 1]

            self._segments.append((x_l, x_r, y_l, y_r, d_l, d_r, r_i))
            self._rational_cubic_splines.append(
                self.P_over_Q(x_l=x_l, x_r=x_r, y_l=y_l, y_r=y_r, d_l=d_l, d_r=d_r, r=r_i)
            )

    def __call__(self, x):
        if isinstance(x, float):
            x = [x]
        output = []
        for x_ in x:
            if self.x[0] > x_:
                output.append(self._rational_cubic_splines[0](x_))
            if self.x[-1] <= x_:
                output.append(self._rational_cubic_splines[-1](x_))
            for i in range(len(self.x) - 1):
                if self.x[i] <= x_ < self.x[i + 1]:
                    output.append(self._rational_cubic_splines[i](x_))
        return np.array(output)

    def derivative(self, x, nu=1):
        x = np.atleast_1d(x).astype(float)
        output = np.empty_like(x)
        for j, x_value in enumerate(x):
            i = np.searchsorted(self.x, x_value, side='right') - 1
            i = np.clip(i, 0, len(self.x) - 2)
            x_l, x_r, f_l, f_r, d_l, d_r, r = self._segments[i]
            h = x_r - x_l
            theta = (x_value - x_l) / h
            one_minus_theta = 1 - theta
            A = r * f_r - h * d_r
            B = r * f_l + h * d_l
            P = (
                f_r * theta**3
                + A * theta**2 * one_minus_theta
                + B * theta * one_minus_theta**2
                + f_l * one_minus_theta**3
            )
            P_1 = (
                3 * f_r * theta**2
                + A * (2 * theta * one_minus_theta - theta**2)
                + B * (one_minus_theta**2 - 2 * theta * one_minus_theta)
                - 3 * f_l * one_minus_theta**2
            )
            Q = 1 + (r - 3) * theta * one_minus_theta
            Q_1 = (r - 3) * (1 - 2 * theta)

            if nu == 1:
                output[j] = (P_1 * Q - P * Q_1) / (Q**2 * h)
            elif nu == 2:
                Delta = (f_r - f_l) / h
                left_gap = Delta - d_l
                right_gap = d_r - Delta
                left_weight = (r - 3) * one_minus_theta**3 + 3 * one_minus_theta - 1
                right_weight = (r - 3) * theta**3 + 3 * theta - 1
                output[j] = 2 * (
                    left_gap * left_weight + right_gap * right_weight
                ) / (h * Q**3)
            else:
                raise ValueError('Поддерживаются только производные nu=1 и nu=2')
        return output

    def P_over_Q(self, x_l: float, x_r: float, y_l: float, y_r: float, d_l: float, d_r: float, r: float):
        return lambda x: self.P(x_l=x_l, x_r=x_r, f_l=y_l, f_r=y_r, d_l=d_l, d_r=d_r, r=r)(x) / self.Q(x_l=x_l, x_r=x_r, r=r)(x)

    def P(self, x_l: float, x_r: float, f_l: float, f_r: float, d_l: float, d_r: float, r: float):
        h = (x_r - x_l)
        theta = lambda x: (x - x_l) / h
        return lambda x: f_r * (theta(x))**3 + (r * f_r - h * d_r) * (theta(x))**2 * (1 - theta(x)) + \
            (r * f_l + h * d_l) * (theta(x)) * (1 - theta(x))**2 + f_l * (1 - theta(x))**3

    def Q(self, x_l: float, x_r: float, r: float):
        h = (x_r - x_l)
        theta = lambda x: (x - x_l) / h
        return lambda x: 1 + (r - 3) * theta(x) * (1 - theta(x))

def r_boundary(d, f, x):
    r = []
    for i in range(len(x) - 1):
        Delta_i = (f[i + 1] - f[i]) / (x[i + 1] - x[i])
        r.append(max((d[i + 1] - d[i]) / (d[i + 1] - Delta_i), (d[i + 1] - d[i]) / (Delta_i - d[i])))
    return r

def d_1_approx(x_1, x_2, x_3, f_1, f_2, f_3):
    Delta_1 = - (f_2 - f_1) / (x_2 - x_1)
    Delta_31 = - (f_3 - f_1) / (x_3 - x_1)
    return - (Delta_1 ** ((x_3 - x_1) / (x_3 - x_2))) * (Delta_31 ** (-(x_2 - x_1) / (x_3 - x_2)))

def d_approx(x_1, x_2, x_3, f_1, f_2, f_3):
    Delta_1 = - (f_2 - f_1) / (x_2 - x_1)
    Delta_2 = - (f_3 - f_2) / (x_3 - x_2)
    return - (Delta_1 ** ((x_3 - x_2) / (x_3 - x_1))) * (Delta_2 ** ((x_2 - x_1) / (x_3 - x_1)))

def d_n_approx(x_n_2, x_n_1, x_n, f_n_2, f_n_1, f_n):
    Delta_n_1 = -(f_n - f_n_1) / (x_n - x_n_1)
    Delta_nn_2 = -(f_n - f_n_2) / (x_n - x_n_2)

    if np.isclose(Delta_n_1, 0) and np.isclose(Delta_nn_2, 0):
        return 0.0

    return -(Delta_n_1 ** ((x_n - x_n_2) / (x_n_1 - x_n_2))) * \
            (Delta_nn_2 ** (-(x_n - x_n_1) / (x_n_1 - x_n_2)))

def slopes_estimation(x, y):
    slopes = []
    for i in range(len(x)):
        if i == 0:
            slopes.append(d_1_approx(x_1=x[i], x_2=x[i+1], x_3=x[i+2], f_1=y[i], f_2=y[i+1], f_3=y[i+2]))
        elif i == len(x) - 1:
            slopes.append(d_n_approx(x_n_2=x[i-2], x_n_1=x[i-1], x_n=x[i], f_n_2=y[i-2], f_n_1=y[i-1], f_n=y[i]))
        else:
            slopes.append(d_approx(x_1=x[i-1], x_2=x[i], x_3=x[i+1], f_1=y[i-1], f_2=y[i], f_3=y[i+1]))

    return slopes
