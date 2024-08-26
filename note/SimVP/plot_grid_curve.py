import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_grid_visual(n, m, grids, margin=0.0, titles=None):
    fig = plt.figure(figsize=(3 * (m + 1), 3 * n))
    gs = gridspec.GridSpec(n, m + 1, wspace=0.04, hspace=0.04)
    for i in range(n):
        ax = plt.subplot(gs[i, 0])
        ax.text(0.5, 0.5, titles[i], fontsize=24, ha='center', va='center', fontname='Times New Roman')
        ax.axis('off')  # 不显示坐标轴
        for j in range(1, m + 1):
            ax = plt.subplot(gs[i, j])
            ax.imshow(grids[i][j-1], cmap='viridis')
            ax.axis('off')
            # if titles != None:
            #     ax.set_title(f'{titles[i]} Timestep {j+1}', fontsize=12)  # 设置标题字体大小
            if i != 0:
                pos = ax.get_position()  # 微调空白间距的细节
                ax.set_position([pos.x0, pos.y0 + margin * i, pos.width, pos.height])  # 向上移动
    plt.savefig('mnist_simvp_curve.pdf', dpi=600, bbox_inches='tight')

if __name__ == '__main__':
    # 1.读取测试数据文件
    # (batch, timesteps, channels, height, weight)
    groudtruth_trues = np.load("./trues.npy", allow_pickle=True)
    print(groudtruth_trues.shape)
    simvp_preds = np.load("./preds.npy", allow_pickle=True)
    # convlstm_preds = np.load("./b1_ConvLSTM_preds.npy", allow_pickle=True)
    # eartherformer_preds = np.load("./b1_Eartherformer_preds.npy", allow_pickle=True)
    # unet_preds = np.load("./b1_UNet_preds.npy", allow_pickle=True)
    
    # 2.设置需要可视化的id
    id = 200
    groudtruth_trues = groudtruth_trues[id, :, 0, :, :]
    simvp_preds = simvp_preds[id, :, 0, :, :]
    # eartherformer_preds = eartherformer_preds[id, :, 0, :, :]
    # unet_preds = unet_preds[id, :, 0, :, :]
    # convlstm_preds = convlstm_preds[id, :, 0, :, :]

    # 3.绘制真实值与预测值的对比图像
    grids = [groudtruth_trues, simvp_preds]
    n = len(grids)  # 3.1 对比图的数量
    m = 10   # 3.2 预测的时间步长
    titles = ['Ground Truth', 'SimVP']
    plot_grid_visual(n, m, grids, titles=titles)