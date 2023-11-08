import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN

from transformer import SquareHFM
import matplotlib.pyplot as plt
import streamlit as st


def get_model(choice_model):
    layers = [2] + 10 * [4 * 15] + [3]
    EPOCHS = 10_000
    LAMDA = 10
    LR = 1e-04
    filepath = "../data/mu0"

    pinn = SquareHFM(layers_list=layers,
                     activation_name="tanh",
                     init_method="xavier_normal",
                     nn_type="vanilla",
                     Re=65035,
                     filepath=filepath,
                     lamda=LAMDA,
                     epochs=EPOCHS,
                     sample_epoch=100,
                     batch_ratio=1,
                     save_name="pinn_transformer",
                     learning_rate=LR,
                     device=torch.device("cpu"),
                     verbose=False)
    # Testing: Prediction after training network
    model_path = f"models/{choice_model}.pth"
    pinn.net_uvp.load_state_dict(torch.load(model_path, map_location=pinn.device), strict=False)
    return pinn


def get_pred(ds):
    x, y = np.meshgrid(np.linspace(0.01, 0.99, 50), np.linspace(0.01, 0.99, 50))
    device = torch.device('cpu')
    in_nums = x.reshape(-1).shape[0]
    data = ds[['x', 'y', 'x-velocity', 'y-velocity', 'pressure']].to_numpy()
    data = np.tile(data, (in_nums, 1, 1))
    z_data = torch.from_numpy(data).float().requires_grad_(True).to(device)

    x_train = torch.tensor(x, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)

    x_data = x_train.reshape(-1, 1).clone().requires_grad_(True).to(pinn.device)
    y_data = y_train.reshape(-1, 1).clone().requires_grad_(True).to(pinn.device)
    u_pred, v_pred, p_pred = pinn.net_uvp(x_data, y_data, z_data)
    return u_pred, v_pred, p_pred

def point_data(path):
    data = pd.read_csv(path, encoding='utf-8', skiprows=11).round({'x': 2, 'y': 2})
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
        sampled_category_df = category_df.sample(n=sample_size, replace=True)
        sampled_dfs.append(sampled_category_df)

    data = pd.concat(sampled_dfs).reset_index(drop=True)
    return data


if __name__ == '__main__':
    st.set_page_config(
        page_title="PINN二维方腔流演示",
        page_icon=":robot:",
        layout='wide'
    )

    st.title("PINN二维方腔流演示")
    choice = st.selectbox("请选择要加载的模型", ("pinn_transformer_10", "pinn_transformer_60"))
    pinn = get_model(choice)

    choice_data = st.selectbox("请选择要加载的数据", ("cavity-0.101", "cavity-0.3", "cavity-0.699"))
    path = f"data/{choice_data}.csv"
    data = point_data(path)
    button = st.button(":point_right:随机生成监督点，并生成方腔流", key="random")

    left_column, right_column = st.columns([3, 5])

    with left_column:
        st.subheader('监督点数据和散点图')
        if button:
            data = data.sample(frac=1).reset_index(drop=True)
            num_samples = np.random.randint(4, 17)
            data = data.sample(n=num_samples)

            st.dataframe(data, use_container_width=True)
            x, y = data['x'].values, data['y'].values
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(x, y)
            fig.tight_layout(pad=1.0)
            st.pyplot(fig)

    with right_column:
        st.subheader('生成方腔流图')
        if button:
            u_pred, v_pred, p_pred = get_pred(data)
            u, v, p = u_pred.data.numpy().reshape(50, 50), v_pred.data.numpy().reshape(50, 50), p_pred.data.numpy().reshape(50, 50)
            x, y = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
            fig, axs = plt.subplots(2, 2, figsize=(7.5, 6), sharey=False)

            N_resolution = 40
            ### Plot u ###
            cp_u = axs[0, 0].contourf(x, y, u, N_resolution)
            fig.colorbar(cp_u)
            cp_u.set_cmap('jet')
            axs[0, 0].set_title('Contours of $u$')
            axs[0, 0].set_xlabel('$x$')
            axs[0, 0].set_ylabel('$y$')

            ### Plot v ###
            cp_v = axs[0, 1].contourf(x, y, v, N_resolution)
            fig.colorbar(cp_v)
            cp_v.set_cmap('jet')
            axs[0, 1].set_title('Contours of $v$')
            axs[0, 1].set_xlabel('$x$')
            axs[0, 1].set_ylabel('$y$')

            ### Plot velocity field ###
            strm = axs[1, 0].streamplot(x, y, u, v, color=v, density=1.5, linewidth=1)
            fig.colorbar(strm.lines)
            strm.lines.set_cmap('jet')
            axs[1, 0].set_title('Velocity stream traces')
            axs[1, 0].set_xlabel('$x$')
            axs[1, 0].set_ylabel('$y$')

            ### Plot p ###
            cp_p = axs[1, 1].contourf(x, y, p,  N_resolution)
            fig.colorbar(cp_p)
            cp_p.set_cmap('jet')
            axs[1, 1].set_title('Contours of $p$')
            axs[1, 1].set_xlabel('$x$')
            axs[1, 1].set_ylabel('$y$')

            fig.tight_layout(pad=1.0)
            st.pyplot(fig)




