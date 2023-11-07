import datetime
import os
import time
import warnings
from pathlib import Path

import pandas as pd
import torch.optim as optim
from sklearn.cluster import DBSCAN
from torch.autograd import Variable

from networks import *
from utilities import *

torch.manual_seed(24)
np.random.seed(24)
torch.set_default_dtype(torch.float32)
warnings.filterwarnings("ignore")


class SquareHFM(nn.Module):
    """2D Hidden Fluid Mechanics class.

    Parameters:
        layers_list (List): Number of input, hidden and output neurons.
        activation_name (str): Type of activation function. Default is `Sine`
        init_method (str): Weight initialization method. Default is `xavier_normal`
        _data : denotes training data (x_data, y_data, t_data, u_data, v_data)
        minmax: ndarray of minimum and maximum value of all training data
        batch_ratio: fraction of batch size for training
        lamda: regularization parameter for tuning PDE equation loss. Default is 1.0
        epochs: Number of epochs
        sample_epoch: Epoch to display training results
        save_name: Save model name. If None, the model name is automatically created from the class.
        verbose: Display the plots of prediction when True. Default is False
    """

    def __init__(self, layers_list, activation_name="sine", init_method="xavier_normal",
                 nn_type="resnet", save_name=None, verbose=False, *,
                 Re, filepath,
                 lamda, epochs, batch_ratio, sample_epoch, learning_rate, device):
        super().__init__()

        self.dim = layers_list[0]
        self.Rex = Re
        self.lamda = lamda
        self.num_epochs = epochs
        self.verbose = verbose
        self.ratio = batch_ratio
        self.lr = learning_rate
        self.interval = sample_epoch
        self.device = device
        datasets = get_datasets(filepath)
        # to_csv(datasets, '../data/train')
        self.point_datas = get_point_data(datasets=datasets, device=device)
        self.eqn_datas = in_data(datasets=datasets, in_nums=5000, device=device)

        if save_name is None:
            self.model_name = self.__class__.__qualname__
        else:
            self.model_name = save_name

        if nn_type == "vanilla":
            self.net_uvp = Neural_Net(layers_list,
                                      activation_name=activation_name,
                                      init_method=init_method)
        elif nn_type == "resnet":
            self.net_uvp = ResNet(layers_list,
                                  activation_name=activation_name,
                                  init_method=init_method)
        elif nn_type == "denseresnet":
            self.net_uvp = DenseResNet(layers_list,
                                       num_res_blocks=5,
                                       num_layers_per_block=2,
                                       fourier_features=True,
                                       tune_beta=True,
                                       m_freqs=100,
                                       sigma=1,
                                       activation_name=activation_name,
                                       init_method=init_method)
        self.net_uvp.model_capacity
        self.optimizer = optim.Adam(self.net_uvp.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=2)

        self.create_dir(f"../logs/{self.model_name}/checkpoint")
        self.create_dir(f"../logs/{self.model_name}/model")
        self.create_dir(f"../logs/{self.model_name}/results")

    def physics_constraints(self, x, y, z, Rex=1000):
        ro = 1
        miu = 1e-2
        u, v, p = self.net_uvp(x, y, z)

        u_x = compute_gradients(u, x)
        u_y = compute_gradients(u, y)
        v_x = compute_gradients(v, x)
        v_y = compute_gradients(v, y)
        p_x = compute_gradients(p, x)
        p_y = compute_gradients(p, y)
        u_xx = compute_gradients(u_x, x)
        u_yy = compute_gradients(u_y, y)
        v_xx = compute_gradients(v_x, x)
        v_yy = compute_gradients(v_y, y)
        omega = v_x - u_y

        e1 = (u * u_x + v * u_y) + (p_x - miu * (u_xx + u_yy)) / ro
        e2 = (u * v_x + v * v_y) + (p_y - miu * (v_xx + v_yy)) / ro
        e3 = u_x + v_y

        return u, v, p, omega, e1, e2, e3

    def loss_fn(self, outputs, targets):

        return nn.MSELoss(reduction="mean")(outputs, targets)

    def data_loss(self, x, y, z):
        u_, v_, p_ = self.net_uvp(x, y, z)

        u_loss = self.loss_fn(u_, z[0][:, 2:3])
        v_loss = self.loss_fn(v_, z[0][:, 3:4])
        p_loss = self.loss_fn(p_, z[0][:, 4:5])

        point_loss = u_loss + v_loss + p_loss

        return point_loss

    def equation_loss(self, x, y, z):
        _, _, _, _, e1, e2, e3 = self.physics_constraints(x, y, z)
        f_zeros = torch.from_numpy(np.zeros((x.shape[0], 1))).float().requires_grad_(True).to(self.device)
        loss_equation = self.loss_fn(e1, f_zeros) + self.loss_fn(e2, f_zeros) + self.loss_fn(e3, f_zeros)

        return loss_equation

    def _train(self):
        choice = input("Resume (y) or New (n): ")
        if choice == "n" or choice == "N":
            print("\nStarting new training ...")

            device = self.device
            pinn_net = self.net_uvp.to(device)
            model_name = self.model_name
            losses = np.empty((0, 2), dtype=float)

            train_loss = []

            start_time = time.time()
            for epoch in range(self.num_epochs):
                self.optimizer.zero_grad()
                loss_eqn = .0
                for (x_eqns, y_eqns, z_eqns) in self.eqn_datas:
                    loss_eqn_cur = self.equation_loss(x_eqns, y_eqns, z_eqns)
                    loss_eqn += loss_eqn_cur

                loss_point = .0
                for (x_point, y_point, z_point) in self.point_datas:
                    loss_point_cur = self.data_loss(x_point, y_point, z_point)
                    loss_point += loss_point_cur

                # loss = loss_u + loss_v + self.lamda * loss_eqn
                loss = loss_point + self.lamda * loss_eqn
                train_loss.append(to_numpy(loss))
                loss.backward()
                self.optimizer.step()

                with torch.autograd.no_grad():

                    loss_eq = to_numpy(loss_eqn).reshape(1, 1)
                    total_loss = to_numpy(loss).reshape(1, 1)
                    lr = self.optimizer.param_groups[0]["lr"]

                if (epoch + 1) % 25 == 0:
                    self.callback(epoch, total_loss, loss_eq, lr)
                all_losses = np.concatenate(
                    [total_loss, loss_eq], axis=1)
                losses = np.append(losses, all_losses, axis=0)

                loss_log = pd.DataFrame(losses)
                loss_log.to_csv(
                    f"../logs/{model_name}/results/losses.csv",
                    index=False,
                    header=["Loss", "Loss Eqns"]
                )
                del loss_log

                if (epoch + 1) % self.interval == 0:
                    state = {
                        "epoch": epoch + 1,
                        "state_dict": pinn_net.state_dict(),
                        "optimizer_dict": self.optimizer.state_dict()
                    }
                    self.save_checkpoint(
                        state, checkpoint_dir=f"../logs/{model_name}/checkpoint"
                    )

                # if epoch % self.interval == 0:
                #     self.exact_and_predict_at_selected_time(self.selected_time, data_stack)
                if (epoch + 1) % 500 == 0:
                    cur = (epoch + 1) // 500
                    self.save_model(model=pinn_net, target_dir=f"../logs/{model_name}/model", cur=str(cur))

            train_loss = np.array(train_loss).mean()
            self.scheduler.step(train_loss)

            elapsed = time.time() - start_time
            self.save_model(model=pinn_net, target_dir=f"../logs/{model_name}/model")

            with open(f"../logs/{model_name}/results/training_metadata.txt", "w") as f:
                f.write(f"{model_name} training metadata generated at {datetime.datetime.now()}\n")
                f.write(f"{'-' * 80}\n")
                f.write(f"Iterations: {self.num_epochs}\n")
                f.write(f"Training epochs: {self.num_epochs}\n")
                f.write(f"Training time: {elapsed / 3600:2.0f}h ({elapsed:.2f}s)\n")

            print(f"\nTraining completed in {elapsed / 3600:^2.0f}h")

        elif choice == "y" or choice == "Y":
            print("\nResume training from last saved checkpoint ...\n")
            self._resume()

    # def continue_train(self):
    #     choice = input("continue train (n): ")
    #     if choice == "n" or choice == "N":
    #         print("\nStarting new training ...")
    #
    #         device = self.device
    #         pinn_net = self.net_uvp.to(device)
    #         model_name = self.model_name + '_continue'
    #
    #         self.create_dir(f"../logs/{model_name}/checkpoint")
    #         self.create_dir(f"../logs/{model_name}/model")
    #         self.create_dir(f"../logs/{model_name}/results")
    #
    #         losses = np.empty((0, 12), dtype=float)
    #
    #         train_loss = []
    #
    #         start_time = time.time()
    #         for epoch in range(self.num_epochs):
    #             self.optimizer.zero_grad()
    #
    #             loss_eqn = self.equation_loss(self.x_eqns, self.y_eqns, self.k_eqns)
    #             loss_point = self.data_loss(self.x_point, self.y_point, self.k_point, self.u_point, self.v_point, self.p_point)
    #
    #             ##### Origin #####
    #             u_origin, v_origin, p_origin = self.net_uvp(self.x_origin_in, self.y_origin_in, self.k_origin_in)
    #             zero = torch.zeros(1, 1).to(device)
    #             loss_origin = self.loss_fn(p_origin, zero)  # pressure is 0 at origin
    #             err_origin = self.relative_error(p_origin, zero)
    #
    #             boundary_num = self.x_up_in.shape[0]
    #             zeros = torch.zeros(boundary_num, 1).to(device)
    #             ones = torch.ones(boundary_num, 1).to(device)
    #
    #             # upper boundary
    #             u_up, v_up, p_up = self.net_uvp(self.x_up_in, self.y_up_in, self.k_up_in)
    #             u_loss_up = self.loss_fn(u_up, self.k_up_in * ones)  # u = 1 at upper boundary
    #             v_loss_up = self.loss_fn(v_up, zeros)  # v = 0 at upper boundary
    #             err_u_up = self.relative_error(u_up, self.k_up_in * ones)
    #
    #             loss_up = u_loss_up + v_loss_up
    #
    #             # lower boundary
    #             u_low, v_low, p_low = self.net_uvp(self.x_low_in, self.y_low_in, self.k_low_in)
    #             u_loss_low = self.loss_fn(u_low, zeros)  # u = 0 at lower boundary
    #             v_loss_low = self.loss_fn(v_low, zeros)  # v = 0 at lower boundary
    #             err_u_low = self.relative_error(u_low, zeros)
    #
    #             loss_low = u_loss_low + v_loss_low
    #
    #             # left boundary
    #             u_left, v_left, p_left = self.net_uvp(self.x_left_in, self.y_left_in, self.k_left_in)
    #             u_loss_left = self.loss_fn(u_left, zeros)  # u = 0 at lower boundary
    #             v_loss_left = self.loss_fn(v_left, zeros)  # v = 0 at lower boundary
    #             err_u_left = self.relative_error(u_left, zeros)
    #
    #             loss_left = u_loss_left + v_loss_left
    #
    #             # right boundary
    #             u_right, v_right, p_right = self.net_uvp(self.x_right_in, self.y_right_in, self.k_right_in)
    #             u_loss_right = self.loss_fn(u_right, zeros)  # u = 0 at lower boundary
    #             v_loss_right = self.loss_fn(v_right, zeros)  # v = 0 at lower boundary
    #             err_u_right = self.relative_error(u_right, zeros)
    #
    #             loss_right = u_loss_right + v_loss_right
    #
    #             # loss = loss_u + loss_v + self.lamda * loss_eqn
    #             loss = loss_origin + loss_up + loss_low + loss_left + loss_right + self.lamda * loss_eqn + loss_point
    #             train_loss.append(to_numpy(loss))
    #             loss.backward()
    #             self.optimizer.step()
    #
    #             with torch.autograd.no_grad():
    #                 loss_origin = to_numpy(loss_origin).reshape(1, 1)
    #                 loss_up = to_numpy(loss_up).reshape(1, 1)
    #                 loss_low = to_numpy(loss_low).reshape(1, 1)
    #                 loss_left = to_numpy(loss_left).reshape(1, 1)
    #                 loss_right = to_numpy(loss_right).reshape(1, 1)
    #
    #                 loss_eq = to_numpy(loss_eqn).reshape(1, 1)
    #                 total_loss = to_numpy(loss).reshape(1, 1)
    #                 lr = self.optimizer.param_groups[0]["lr"]
    #
    #                 err_origin = to_numpy(err_origin).reshape(1, 1)
    #                 err_up = to_numpy(err_u_up).reshape(1, 1)
    #                 err_low = to_numpy(err_u_low).reshape(1, 1)
    #                 err_left = to_numpy(err_u_left).reshape(1, 1)
    #                 err_right = to_numpy(err_u_right).reshape(1, 1)
    #             if (epoch + 1) % 50 == 0:
    #                 self.callback(epoch, total_loss, loss_origin, loss_up, loss_low, loss_left, loss_right, loss_eq,
    #                               err_origin, err_up, err_low, err_left, err_right, lr)
    #             all_losses = np.concatenate(
    #                 [total_loss, loss_origin, loss_up, loss_low, loss_left, loss_right, loss_eq, err_origin, err_up,
    #                  err_low, err_left, err_right], axis=1)
    #             losses = np.append(losses, all_losses, axis=0)
    #
    #             loss_log = pd.DataFrame(losses)
    #             loss_log.to_csv(
    #                 f"../logs/{model_name}/results/losses.csv",
    #                 index=False,
    #                 header=["Loss", "Loss Origin", "Loss Up", "Loss Low", "Loss Left", "Loss Right", "Loss Eqns",
    #                         "Rel_Origin", "Rel_Up", "Rel_Low", "Rel_Left", "Rel_Right"]
    #             )
    #             del loss_log
    #
    #             if (epoch + 1) % self.interval == 0:
    #                 state = {
    #                     "epoch": epoch + 1,
    #                     "state_dict": pinn_net.state_dict(),
    #                     "optimizer_dict": self.optimizer.state_dict()
    #                 }
    #                 self.save_checkpoint(
    #                     state, checkpoint_dir=f"../logs/{model_name}/checkpoint"
    #                 )
    #
    #             # if epoch % self.interval == 0:
    #             #     self.exact_and_predict_at_selected_time(self.selected_time, data_stack)
    #             if (epoch + 1) % 500 == 0:
    #                 cur = (epoch + 1) // 500
    #                 self.save_model(model=pinn_net, target_dir=f"../logs/{model_name}/model", cur=str(cur))
    #
    #         train_loss = np.array(train_loss).mean()
    #         self.scheduler.step(train_loss)
    #
    #         elapsed = time.time() - start_time
    #         self.save_model(model=pinn_net, target_dir=f"../logs/{model_name}/model")
    #
    #         with open(f"../logs/{model_name}/results/training_metadata.txt", "w") as f:
    #             f.write(f"{model_name} training metadata generated at {datetime.datetime.now()}\n")
    #             f.write(f"{'-' * 80}\n")
    #             f.write(f"Iterations: {self.num_epochs}\n")
    #             f.write(f"Training epochs: {self.num_epochs}\n")
    #             f.write(f"Training time: {elapsed / 3600:2.0f}h ({elapsed:.2f}s)\n")
    #
    #         print(f"\nTraining completed in {elapsed / 3600:^2.0f}h")
    #
    #     elif choice == "y" or choice == "Y":
    #         print("\nResume training from last saved checkpoint ...\n")
    #         self._resume()
    @staticmethod
    def save_checkpoint(model_state, checkpoint_dir):
        ckpt = os.path.join(checkpoint_dir, "checkpoint.pth")
        torch.save(model_state, ckpt)

    @staticmethod
    def callback(epoch, total_loss, loss_eq, lr):
        info = f"Epoch: {epoch + 1:<4d}  " + \
               f"Loss: {total_loss.item():.2e}  " + \
               f"Loss Eqns: {loss_eq.item():.2e}  " + \
               f"LR: {lr:.2e}"
        print(info)

    def save_model(self, model, target_dir, cur="end"):
        model_path = os.path.join(target_dir, f"{self.model_name.lower()}_{cur}.pth")
        torch.save(model.state_dict(), model_path)

    def create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    # def relative_error(self, exact, pred):
    #     return torch.sqrt(torch.mean(torch.square(exact - pred))) / \
    #         torch.sqrt(torch.mean(torch.square(exact)))

    def relative_error(self, y_pred, y_true):
        error = torch.abs((y_true - y_pred) / torch.clamp(torch.abs(y_true), min=1e-8)) * 100.0
        return torch.mean(error)


def my_read_data_portion():
    minmax_value = np.empty((2, 2))
    minmax_value[0, :] = 0
    minmax_value[1, :] = 1
    minmax_value = torch.tensor(minmax_value, dtype=torch.float32)
    return minmax_value


def generate_multi_velocity(x, y, k=None):
    x, y = x.reshape(-1), y.reshape(-1)
    if not k:
        k = np.array(
            [0.101, 0.14, 0.16, 0.175, 0.203, 0.246, 0.249, 0.297, 0.3, 0.343, 0.359, 0.387, 0.392, 0.393, 0.443,
             0.471, 0.497, 0.5, 0.608, 0.626, 0.699, 0.738, 0.743, 0.812, 0.837, 0.868, 0.89, 0.9, 0.902, 0.915,
             0.951, 0.972, 0.977, 1.0])[::10]
    m, n = len(x), len(k)
    x = np.tile(x, (1, n))
    y = np.tile(y, (1, n))
    k = np.tile(k, (m, 1)).T.reshape(-1)
    return x, y, k


def get_point_data(datasets, device):
    ans = []
    for ds in datasets:
        data = ds[['x', 'y', 'x-velocity', 'y-velocity', 'pressure']].to_numpy()
        x = ds["x"].values
        y = ds["y"].values
        x = Variable(torch.from_numpy(np.reshape(x, (-1, 1))).float(), requires_grad=True).to(device)
        y = Variable(torch.from_numpy(np.reshape(y, (-1, 1))).float(), requires_grad=True).to(device)
        data = torch.from_numpy(data).float().requires_grad_(True).to(device)
        data = data.repeat(data.shape[0], 1, 1)
        ans.append((x, y, data))
    return ans


def to_csv(datasets, path):
    p = Path(path)
    for i, data in enumerate(datasets):
        data.to_csv(p / f'{i}.csv', encoding='utf-8', index=False)


def get_datasets(path):
    p = Path(path)
    files = p.glob('*.csv')
    datasets = []
    num_datasets = 6
    for file in list(files)[::4]:
        for i in range(num_datasets):
            df = point_data(file)
            df = df.sample(frac=1, random_state=4).reset_index(drop=True)
            num_samples = np.random.randint(4, 17)
            random_samples = df.sample(n=num_samples, random_state=4)
            datasets.append(random_samples)
    print(len(datasets))
    return datasets


def point_data(filepath, k=None):
    p = Path(filepath)
    data = pd.read_csv(p, encoding='utf-8', skiprows=11).round({'x': 2, 'y': 2})
    data = data.sort_values(by=['y', 'x'], ascending=[True, True]).reset_index(drop=True)

    dbscan = DBSCAN(eps=0.05, min_samples=20)
    dbscan.fit(data[['x', 'y', 'x-velocity', 'y-velocity']])
    labels = dbscan.labels_
    data['label'] = labels

    category_counts = data['label'].value_counts()

    total_samples = data.shape[0]
    sample_ratios = (category_counts / total_samples).tolist()

    sampled_dfs = []
    for category, ratio in zip(category_counts.index, sample_ratios):
        category_df = data[data['label'] == category]
        sample_size = min(int(ratio * total_samples * 0.5), 100)
        sampled_category_df = category_df.sample(n=sample_size, replace=True, random_state=4)
        sampled_dfs.append(sampled_category_df)

    data = pd.concat(sampled_dfs).reset_index(drop=True)
    return data


def in_data(datasets, in_nums=5000, device=torch.device('cpu')):
    ans = []
    for ds in datasets:
        data = ds[['x', 'y', 'x-velocity', 'y-velocity', 'pressure']].to_numpy()
        res_pts = lhs(in_nums, 2)
        x_res = res_pts[:, 0]
        y_res = res_pts[:, 1]
        x_res_in = Variable(torch.from_numpy(np.reshape(x_res, (-1, 1))).float(), requires_grad=True).to(device)
        y_res_in = Variable(torch.from_numpy(np.reshape(y_res, (-1, 1))).float(), requires_grad=True).to(device)
        data = torch.from_numpy(data).float().requires_grad_(True).to(device)
        data = data.repeat(in_nums, 1, 1)
        ans.append((x_res_in, y_res_in, data))
    return ans


def border_data(boundary_num=50, device=torch.device('cpu'), k=None):
    # upper boundary
    x_up = np.linspace(0., 1., boundary_num)
    y_up = np.ones(boundary_num)
    x_up, y_up, k_up = generate_multi_velocity(x_up, y_up, k)
    x_up_in = Variable(torch.from_numpy(np.reshape(x_up, (-1, 1))).float(), requires_grad=True).to(device)
    y_up_in = Variable(torch.from_numpy(np.reshape(y_up, (-1, 1))).float(), requires_grad=True).to(device)
    k_up_in = Variable(torch.from_numpy(np.reshape(k_up, (-1, 1))).float(), requires_grad=True).to(device)

    # lower boundary
    x_low = np.linspace(0., 1., boundary_num)
    y_low = np.zeros(boundary_num)
    x_low, y_low, k_low = generate_multi_velocity(x_low, y_low, k)

    x_low_in = Variable(torch.from_numpy(np.reshape(x_low, (-1, 1))).float(), requires_grad=True).to(device)
    y_low_in = Variable(torch.from_numpy(np.reshape(y_low, (-1, 1))).float(), requires_grad=True).to(device)
    k_low_in = Variable(torch.from_numpy(np.reshape(k_low, (-1, 1))).float(), requires_grad=True).to(device)

    # left boundary
    x_left = np.zeros(boundary_num)
    y_left = np.linspace(0., 1., boundary_num)
    x_left, y_left, k_left = generate_multi_velocity(x_left, y_left, k)

    x_left_in = Variable(torch.from_numpy(np.reshape(x_left, (-1, 1))).float(), requires_grad=True).to(device)
    y_left_in = Variable(torch.from_numpy(np.reshape(y_left, (-1, 1))).float(), requires_grad=True).to(device)
    k_left_in = Variable(torch.from_numpy(np.reshape(k_left, (-1, 1))).float(), requires_grad=True).to(device)

    # right boundary
    x_right = np.ones(boundary_num)
    y_right = np.linspace(0., 1., boundary_num)
    x_right, y_right, k_right = generate_multi_velocity(x_right, y_right, k)

    x_right_in = Variable(torch.from_numpy(np.reshape(x_right, (-1, 1))).float(), requires_grad=True).to(device)
    y_right_in = Variable(torch.from_numpy(np.reshape(y_right, (-1, 1))).float(), requires_grad=True).to(device)
    k_right_in = Variable(torch.from_numpy(np.reshape(k_right, (-1, 1))).float(), requires_grad=True).to(device)

    return x_up_in, y_up_in, k_up_in, x_low_in, y_low_in, k_low_in, x_left_in, y_left_in, k_left_in, x_right_in, y_right_in, k_right_in


def origin_data(device, k):
    x_origin = np.array([[0.]])
    y_origin = np.array([[0.]])

    x_origin, y_origin, k_origin = generate_multi_velocity(x_origin, y_origin, k)
    x_origin_in = Variable(torch.from_numpy(np.reshape(x_origin, (-1, 1))).float(), requires_grad=True).to(device)
    y_origin_in = Variable(torch.from_numpy(np.reshape(y_origin, (-1, 1))).float(), requires_grad=True).to(device)
    k_origin_in = Variable(torch.from_numpy(np.reshape(k_origin, (-1, 1))).float(), requires_grad=True).to(device)
    return x_origin_in, y_origin_in, k_origin_in


if __name__ == "__main__":
    layers = [2] + 10 * [4 * 5] + [3]
    EPOCHS = 10_000
    LAMDA = 10
    LR = 1e-04

    pinn = SquareHFM(layers_list=layers,
                     activation_name="swish",
                     init_method="xavier_normal",
                     nn_type="vanilla",
                     Re=65035,
                     filepath='../data/mu0',
                     lamda=LAMDA,
                     epochs=EPOCHS,
                     sample_epoch=100,
                     batch_ratio=0.5,
                     save_name="Cylinder2D_Wake",
                     learning_rate=LR,
                     device=torch.device("cpu"),
                     verbose=False)

    pinn._train()
