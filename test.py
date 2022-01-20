import numpy as np

N = 100

x = np.random.rand(N)

def normalize(x):
    return x / x.std(ddof=1)

def jacob_norm(x):
    std = x.std(ddof=1)
    xx = np.outer(x, x - x.mean())
    return np.eye(N) / std - xx / ((N - 1) * std ** 3)

def differentiate(f, x, eps=1e-12):
    ret = np.empty((x.size, x.size), dtype=np.float64)
    for j in range(x.size):
        df_dxj = f(x)
        x[j] -= eps
        df_dxj -= f(x)
        x[j] += eps
        df_dxj /= eps
        ret[:, j] = df_dxj
    return ret


def log_jacobian(x):
    mean = np.mean(x)
    std = x.std(ddof=1)
    a = 1. / std
    u = x
    v = -(x - mean) / ((std ** 3) * (N - 1))
    uv = u * v
    d = a / uv
    log_det = np.sum(np.log(np.abs(d)))
    log_det += np.log(np.abs(1. + np.sum(1. / d)))
    log_det += np.sum(np.log(np.abs(u * v)))
    return log_det


theoretical = jacob_norm(x)
empirical = differentiate(normalize, x)
print(f'{np.mean(np.abs(theoretical - empirical)):e}')

theor_log_jac = log_jacobian(x)
empir_log_jac = np.linalg.slogdet(jacob_norm(x))[1]
print(theor_log_jac, empir_log_jac)
