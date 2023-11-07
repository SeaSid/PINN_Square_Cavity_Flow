import numpy as np
import scipy.io
import torch


def read_data(filepath):
    data = scipy.io.loadmat(filepath)

    U_star = data["u_star"]
    V_star = data["v_star"]
    X_star = data["x_star"]
    Y_star = data["y_star"]
    T_star = data["t"]

    X_star = X_star.reshape(-1, 1)
    Y_star = Y_star.reshape(-1, 1)

    Nx = X_star.shape[0]
    Nt = T_star.shape[0]

    X = np.tile(X_star, (1, Nt))
    Y = np.tile(Y_star, (1, Nt))
    T = np.tile(T_star, (1, Nx)).T
    U = U_star
    V = V_star

    x = X.ravel().reshape(-1, 1)
    y = Y.ravel().reshape(-1, 1)
    t = T.ravel().reshape(-1, 1)
    u = U.ravel().reshape(-1, 1)
    v = V.ravel().reshape(-1, 1)

    temp = np.concatenate((x, y, t, u, v), 1)
    minmax_value = np.empty((2, 5))
    minmax_value[0, :] = np.min(temp, axis=0)
    minmax_value[1, :] = np.max(temp, axis=0)

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    minmax_value = torch.tensor(minmax_value, dtype=torch.float32)

    return x, y, t, u, v, minmax_value


def read_data_portion(filepath, percent):
    np.random.seed(11)
    data = scipy.io.loadmat(filepath)

    U_star = data["u_star"]
    V_star = data["v_star"]
    X_star = data["x_star"]
    Y_star = data["y_star"]
    T_star = data["t"]

    X_star = X_star.reshape(-1, 1)
    Y_star = Y_star.reshape(-1, 1)

    Nx = X_star.shape[0]
    Nt = T_star.shape[0]

    X = np.tile(X_star, (1, Nt))
    Y = np.tile(Y_star, (1, Nt))
    T = np.tile(T_star, (1, Nx)).T

    indices_t = np.random.choice(Nt, int(percent * Nt), replace=False)
    indices_t = np.sort(indices_t)

    t = T[:, indices_t].reshape(-1, 1)
    x = X[:, indices_t].reshape(-1, 1)
    y = Y[:, indices_t].reshape(-1, 1)
    u = U_star[:, indices_t].reshape(-1, 1)
    v = V_star[:, indices_t].reshape(-1, 1)

    temp = np.concatenate((x, y, t, u, v), 1)
    minmax_value = np.empty((2, 5))
    minmax_value[0, :] = np.min(temp, axis=0)
    minmax_value[1, :] = np.max(temp, axis=0)

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    minmax_value = torch.tensor(minmax_value, dtype=torch.float32)

    return x, y, t, u, v, minmax_value


def data_split(df, a_min, a_max, b_min, b_max):
    conditions = ((df['x'] >= a_min) & (df['x'] <= a_max) | (df['x'].isna().all())) & \
                 ((df['y'] >= b_min) & (df['y'] <= b_max) | (df['y'].isna().all()))
    if conditions.sum() == 0:
        raise ValueError('偏微分方程数据按值裁剪后，剩余数据为0！')
    return df[conditions]

def my_read_data_portion(data):
    px = data['dp-dx'].values.reshape(-1, 1)
    py = data['dp-dy'].values.reshape(-1, 1)
    u = data['u'].values.reshape(-1, 1)
    ux = data['du-dx'].values.reshape(-1, 1)
    uy = data['du-dy'].values.reshape(-1, 1)
    v = data['v'].values.reshape(-1, 1)
    vx = data['dv-dx'].values.reshape(-1, 1)
    vy = data['dv-dy'].values.reshape(-1, 1)
    txx = data['tau_xx'].values.reshape(-1, 1)
    txy = data['tau_xy'].values.reshape(-1, 1)
    tyy = data['tau_yy'].values.reshape(-1, 1)

    px = torch.tensor(px, dtype=torch.float32)
    py = torch.tensor(py, dtype=torch.float32)
    ux = torch.tensor(ux, dtype=torch.float32)
    uy = torch.tensor(uy, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    vx = torch.tensor(vx, dtype=torch.float32)
    vy = torch.tensor(vy, dtype=torch.float32)
    txx = torch.tensor(txx, dtype=torch.float32)
    txy = torch.tensor(txy, dtype=torch.float32)
    tyy = torch.tensor(tyy, dtype=torch.float32)

    miut = (1*(- tyy/(2*vy)) + 2*(- tyy/(2*vy)) + 1*(- txy/(uy+vx)))/4

    return px, py, u, ux, uy, v, vx, vy, txx, txy, tyy, miut


def my_compute_gradients(y, x):
    grad = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y),retain_graph=True, create_graph=True)[0]
    grad2 = torch.autograd.grad(outputs=grad, inputs=x, grad_outputs=torch.ones_like(grad), retain_graph=True)[0]
    return grad.data, grad2

def compute_gradients(Y, x):
    dummy = torch.ones_like(Y, requires_grad=True)
    G = torch.autograd.grad(Y, x, grad_outputs=dummy, create_graph=True)[0]
    Y_x = torch.autograd.grad(G, dummy, grad_outputs=torch.ones_like(G), create_graph=True)[0]
    return Y_x

def generate_eqn_data(lower_bound, upper_bound, samples, num_points):
    eqn_points = lower_bound + (upper_bound - lower_bound) * lhs(num_points, samples)
    perm = np.random.permutation(eqn_points.shape[0])
    new_points = eqn_points[perm, :]
    return torch.from_numpy(new_points).float()

def lhs(samples, dimensions):
    lhs = np.zeros((samples, dimensions))

    # 生成等间隔的区间
    intervals = np.linspace(0, 1, samples+1)

    # 针对每个维度，填充LHS矩阵
    for i in range(dimensions):
        column_intervals = np.random.permutation(intervals[:-1])
        column_samples = np.random.uniform(column_intervals, intervals[1:])
        lhs[:, i] = column_samples

    return lhs

def to_numpy(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs.detach().cpu().numpy()
    elif isinstance(inputs, np.ndarray):
        return inputs
    else:
        raise TypeError("Unknown input type! Expected torch.Tensor or np.ndarray, but got {}".format(
            type(inputs))
        )


def gradient_velocity_2D(u, v, x, y):
    u_x = compute_gradients(u, x)
    v_x = compute_gradients(v, x)
    u_y = compute_gradients(u, y)
    v_y = compute_gradients(v, y)
    return u_x, v_x, u_y, v_y


def strain_rate_2D(u, v, x, y):
    u_x, v_x, u_y, v_y = gradient_velocity_2D(u, v, x, y)
    return u_x, 0.5 * (v_x + u_y), v_y


class TorchMinMaxScaler:
    """MinMax Scaler

    Transforms data to range [-1, 1]

    Returns:
        A tensor with scaled features
    """

    def __init__(self):
        self.x_max = None
        self.x_min = None

    def fit(self, x):
        self.x_max = x.max(dim=0, keepdim=True)[0]
        self.x_min = x.min(dim=0, keepdim=True)[0]

    def transform(self, x):
        x.sub_(self.x_min).div_(self.x_max - self.x_min)
        x.mul_(2).sub_(1)
        return x

    def inverse_transform(self, x):
        x.add_(1).div_(2)
        x.mul_(self.x_max - self.x_min).add_(self.x_min)
        return x

    fit_transform = transform


if __name__ == '__main__':
    lhs(300, 3)
    print(1)
