import argparse
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score

uct_abb_dict = {'MP': 'Max Projected Probability',
                'UM': 'Uncertainty Mass',
                'DE': 'Differential Entropy',
                'MI': 'Mutual Information'}


def parse_args():
    parser = argparse.ArgumentParser(description='Draw histogram')
    parser.add_argument('--dir_name', default='cifar10-medl-test-draw')
    parser.add_argument('--file_name', default='alpha0_id_CIFAR10_ood_CIFAR100.npy')
    parser.add_argument('--uncertainty', default='UM', choices=['MP', 'UM', 'DE', 'MI'],
                        help='the uncertainty estimation method')
    parser.add_argument('--ind_data', default='CIFAR10', choices=['CIFAR10', 'MNIST'], help='the split file of in-distribution testing data')
    parser.add_argument('--ood_data', default='CIFAR100', choices=['SVHN', 'CIFAR100', 'KMNIST', 'FMNIST'],
                        help='the split file of out-of-distribution testing data')
    parser.add_argument('--model', default='EDL', choices=['EDL', 'IEDL', 'REDL', 'FREDL'])
    parser.add_argument('--result_prefix', default='temp/temp.png', help='result file prefix')
    # parser.add_argument('--aupr', default=80.0, type=float)
    args = parser.parse_args()
    return args


def plot_by_uncertainty(result_file_prefix, uncertainty):
    font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'  # 修改为字体文件的实际路径
    prop = fm.FontProperties(fname=font_path, size=15)

    score_norm_file = f"{result_file_prefix}_scores_norm.npy"
    score_file = f"{result_file_prefix}_scores.npy"
    label_file = f"{result_file_prefix}_labels.npy"
    assert os.path.exists(score_norm_file), 'result file not exists! %s' % (score_norm_file)
    assert os.path.exists(score_file), 'result file not exists! %s' % (score_file)
    assert os.path.exists(label_file), 'result file not exists! %s' % (label_file)
    score_norm = np.load(score_norm_file, allow_pickle=True)

    # with CIFAR10 as ID, the first 10000 numbers are ID data
    id_score = score_norm[:10000]  # (N1,)
    ood_score = score_norm[10000:]  # (N2,)

    scores = np.load(score_file, allow_pickle=True)
    labels = np.load(label_file, allow_pickle=True)
    aupr = average_precision_score(labels, scores) * 100

    # visualize
    fig = plt.figure(figsize=(5, 4))  # (w, h)
    # plt.rcParams["font.family"] = "Arial"  # Times New Roman
    # data_label = 'SVHN' if args.ood_data == 'SVHN' else 'CIFAR100'
    data_label = args.ood_data
    plt.hist([id_score, ood_score], 60,
             density=True, histtype='bar', color=['blue', 'red'],
             label=['In-Distribution (%s)' % (args.ind_data), 'Out-of-Distribution (%s)' % (data_label)])
    plt.legend(prop=prop)
    plt.text(0.4, 10, 'AUPR = %.2lf' % (aupr), fontsize=20, fontproperties=prop)
    plt.xlabel(uct_abb_dict[uncertainty], fontsize=20, fontproperties=prop)
    plt.ylabel('Density', fontsize=20, fontproperties=prop)
    plt.xlim(0, 1.01)
    plt.ylim(0, 15.01)

    # 设置坐标轴刻度的字体和大小
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=10)  # 设置坐标轴刻度数字的大小
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(prop)  # 设置坐标轴刻度数字的字体

    plt.tight_layout()

    result_dir = os.path.dirname(args.result_prefix)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # save the figure
    plt.savefig(os.path.join(result_file_prefix + '_distribution.pdf'), bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)


if __name__ == '__main__':

    args = parse_args()
    result_file_prefix = f'../results/{args.dir_name}/{args.file_name}'
    plot_by_uncertainty(result_file_prefix, uncertainty=args.uncertainty)

