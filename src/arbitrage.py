import numpy as np
import cvxpy as cp

def vertical_spread_arbitrage(c, k, discount_factor, tolerance=1e-9):
    slopes = np.diff(c) / np.diff(k)
    return (slopes > tolerance) | (slopes < -discount_factor - tolerance)


def convexity_arbitrage(c, k, tolerance=1e-9):
    slopes = np.diff(c) / np.diff(k)
    return np.diff(slopes) < -tolerance

def fit_convex_prices(smile):
    smile = (smile.sort_values('strike').copy())

    K = smile['strike'].to_numpy(dtype=float)
    F = smile['forward'].to_numpy(dtype=float)
    B = smile['discount_factor'].to_numpy(dtype=float)

    is_call = (smile['option_kind'].eq('Call').to_numpy())

    parity_shift = B * (F - K)

    call_bid = np.where(
        is_call,
        smile['bid'],
        smile['bid'] + parity_shift
    )

    call_ask = np.where(
        is_call,
        smile['ask'],
        smile['ask'] + parity_shift
    )

    call_mid = (call_bid + call_ask) / 2

    call_lower_bound = B * np.maximum(F - K, 0)
    call_upper_bound = B * F

    lower_bound = np.maximum(
        call_bid,
        call_lower_bound
    )

    upper_bound = np.minimum(
        call_ask,
        call_upper_bound
    )

    if np.any(lower_bound > upper_bound):
        raise ValueError(
            'Для данной экспирации bid-ask несовместим '
            'с границами цены опциона'
        )

    fitted_call = cp.Variable(len(smile))

    slopes = (fitted_call[1:] - fitted_call[:-1]) / (K[1:] - K[:-1])

    spread = np.maximum(call_ask - call_bid, 1e-6)

    objective = cp.Minimize(
        cp.sum_squares(
            cp.multiply(1 / spread, fitted_call - call_mid)
        )
    )

    F0 = F[0]
    B0 = B[0]

    x = K / F0
    normalized_call = fitted_call / (B0 * F0)

    normalized_slopes = (
        (normalized_call[1:] - normalized_call[:-1])
        / (x[1:] - x[:-1])
    )

    dx_mid = (x[2:] - x[:-2]) / 2
    curvature_floor = 1e-4

    constraints = [
        fitted_call >= lower_bound,
        fitted_call <= upper_bound,
        normalized_slopes >= -1,
        normalized_slopes <= 0,
        normalized_slopes[1:] >= normalized_slopes[:-1] + curvature_floor * dx_mid
    ]

    problem = cp.Problem(
        objective,
        constraints
    )

    problem.solve(solver=cp.CLARABEL, verbose=False)

    if problem.status not in {
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE
    }:
        raise RuntimeError(
            'Не удалось найти безарбитражные цены: '
            + problem.status
        )

    fitted_call_value = np.asarray(
        fitted_call.value
    ).reshape(-1)

    smile['call_midf'] = fitted_call_value


    smile['midf'] = np.where(
        is_call,
        fitted_call_value,
        fitted_call_value - parity_shift
    )

    smile['call_bid'] = call_bid
    smile['call_ask'] = call_ask
    smile['call_mid'] = call_mid

    return smile

def surface_arbitrage_report(normalized_calls, moneyness_grid, tolerance=1e-10):
    normalized_calls = np.asarray(normalized_calls)
    moneyness_grid = np.asarray(moneyness_grid)

    if normalized_calls.ndim != 2:
        raise ValueError('normalized_calls должна быть двумерной матрицей')

    if normalized_calls.shape[1] != len(moneyness_grid):
        raise ValueError('Размер сетки moneyness не совпадает с размером поверхности')

    lower_price_bound = np.maximum(1 - moneyness_grid, 0)[None, :]
    upper_price_bound = np.ones((1, len(moneyness_grid)))

    lower_price_violations = normalized_calls < lower_price_bound - tolerance
    upper_price_violations = normalized_calls > upper_price_bound + tolerance
    calendar_violations = np.diff(normalized_calls, axis=0) < -tolerance

    strike_slopes = np.diff(normalized_calls, axis=1) / np.diff(moneyness_grid)[None, :]

    upper_slope_violations = strike_slopes > tolerance
    lower_slope_violations = strike_slopes < -1 - tolerance
    butterfly_violations = np.diff(strike_slopes, axis=1) < -tolerance

    report = {
        'lower_price_violations': int(lower_price_violations.sum()),
        'upper_price_violations': int(upper_price_violations.sum()),
        'calendar_violations': int(calendar_violations.sum()),
        'upper_slope_violations': int(upper_slope_violations.sum()),
        'lower_slope_violations': int(lower_slope_violations.sum()),
        'butterfly_violations': int(butterfly_violations.sum()),
        'minimum_slope': float(strike_slopes.min()),
        'maximum_slope': float(strike_slopes.max())
    }

    print('Нарушения нижней границы цены:', report['lower_price_violations'])
    print('Нарушения верхней границы цены:', report['upper_price_violations'])
    print("Нарушения calendar arbitrage:", report["calendar_violations"])
    print("Нарушения верхней границы наклона:", report["upper_slope_violations"])
    print("Нарушения нижней границы наклона:", report["lower_slope_violations"])
    print("Нарушения butterfly arbitrage:", report["butterfly_violations"])
    print('Диапазон наклона:', report['minimum_slope'], report['maximum_slope'])

    return report