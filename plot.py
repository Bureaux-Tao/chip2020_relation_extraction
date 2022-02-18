import pandas as pd
from matplotlib import pyplot as plt

from path import fig_path, log_path, f1_report_path


def train_plot(history, epoch):
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.dpi'] = 360  # 图片像素
    plt.rcParams['figure.dpi'] = 360  # 分辨率
    train_process = pd.DataFrame(history)
    
    # 损失函数
    plt.plot(epoch, train_process.loss, marker = '^', markevery = 5, color = 'k', label = "Train loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(fig_path + "/train_loss.png")
    plt.show()
    
    train_process.to_csv(log_path)


def f1_plot(data):
    pd_f1 = pd.DataFrame(data)
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.dpi'] = 360  # 图片像素
    plt.rcParams['figure.dpi'] = 360  # 分辨率
    
    plt.plot(pd_f1.epoch, pd_f1.f1, marker = '^', markevery = 5, color = 'k', label = "Val f1")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("f1")
    plt.legend()
    plt.savefig(fig_path + '/val_f1.png')
    plt.show()
    
    pd_f1.to_csv(f1_report_path, encoding = 'utf-8')
