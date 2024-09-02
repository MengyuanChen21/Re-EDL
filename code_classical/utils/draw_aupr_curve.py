import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, roc_curve
import matplotlib.font_manager as fm


if __name__ == '__main__':

    # AUPR_dict = {"max_prob": [82.30, 85.11, 85.00, 87.84],
    #              "alpha0": [82.32, 84.97, 85.00, 89.94],
    #              "diff_ent": [82.30, 85.12, 85.01, 89.34],
    #              "mi": [82.32, 84.98, 85.00, 89.89]}
    # 设置字体路径
    font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'  # 修改为字体文件的实际路径
    prop = fm.FontProperties(fname=font_path, size=15)

    for uct_name in ["max_prob", "alpha0", "diff_ent", "mi"]:
        labels1 = np.load(f"../results/cifar10-edl-test-draw/{uct_name}_id_CIFAR10_ood_SVHN_labels.npy")
        scores1 = np.load(f"../results/cifar10-edl-test-draw/{uct_name}_id_CIFAR10_ood_SVHN_scores.npy")
        labels2 = np.load(f"../results/cifar10-iedl-test-draw/{uct_name}_id_CIFAR10_ood_SVHN_labels.npy")
        scores2 = np.load(f"../results/cifar10-iedl-test-draw/{uct_name}_id_CIFAR10_ood_SVHN_scores.npy")
        labels3 = np.load(f"../results/cifar10-iclr-test-draw/{uct_name}_id_CIFAR10_ood_SVHN_labels.npy")
        scores3 = np.load(f"../results/cifar10-iclr-test-draw/{uct_name}_id_CIFAR10_ood_SVHN_scores.npy")
        labels4 = np.load(f"../results/cifar10-fredl-test-draw/{uct_name}_id_CIFAR10_ood_SVHN_labels.npy")
        scores4 = np.load(f"../results/cifar10-fredl-test-draw/{uct_name}_id_CIFAR10_ood_SVHN_scores.npy")

        # 计算精确度-召回率曲线和平均精确度
        precision1, recall1, _ = precision_recall_curve(labels1, scores1)
        average_precision1 = average_precision_score(labels1, scores1) * 100
        precision2, recall2, _ = precision_recall_curve(labels2, scores2)
        average_precision2 = average_precision_score(labels2, scores2) * 100
        precision3, recall3, _ = precision_recall_curve(labels3, scores3)
        average_precision3 = average_precision_score(labels3, scores3) * 100
        precision4, recall4, _ = precision_recall_curve(labels4, scores4)
        average_precision4 = average_precision_score(labels4, scores4) * 100

        # 绘制四条AUPR曲线在同一张图中
        fig = plt.figure(figsize=(5, 4), dpi=300)
        # plt.plot(recall1, precision1, label=f'EDL (AUPR = {AUPR_dict[uct_name][0]:.2f})')
        # plt.plot(recall2, precision2, label=f'I-EDL (AUPR = {AUPR_dict[uct_name][1]:.2f})')
        # plt.plot(recall3, precision3, label=f'R-EDL (AUPR = {AUPR_dict[uct_name][2]:.2f})')
        # plt.plot(recall4, precision4, label=f'Re-EDL (AUPR = {AUPR_dict[uct_name][3]:.2f})')
        plt.plot(recall1, precision1, label=f'EDL AUPR = {average_precision1:.2f}', linewidth=2)
        plt.plot(recall2, precision2, label=f'I-EDL AUPR = {average_precision2:.2f}', linewidth=2)
        plt.plot(recall3, precision3, label=f'R-EDL AUPR = {average_precision3:.2f}', linewidth=2)
        plt.plot(recall4, precision4, label=f'Re-EDL AUPR = {average_precision4:.2f}', linewidth=2)

        # 添加图例和标签
        plt.grid(True)
        plt.xlabel('Recall', fontsize=20, fontproperties=prop)
        plt.ylabel('Precision', fontsize=20, fontproperties=prop)
        # plt.title('Precision-Recall Curves')
        plt.legend(loc='lower left', prop=prop)

        # 设置坐标轴刻度的字体和大小
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=10)  # 设置坐标轴刻度数字的大小
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(prop)  # 设置坐标轴刻度数字的字体

        plt.tight_layout()
        plt.savefig(f"/data_SSD1/cmy/REDL-TPAMI/code_classical/results/aupr/{uct_name}.pdf", bbox_inches='tight', dpi=250, pad_inches=0.0)

        # 计算ROC曲线
        fpr1, tpr1, _ = roc_curve(labels1, scores1)
        fpr2, tpr2, _ = roc_curve(labels2, scores2)
        fpr3, tpr3, _ = roc_curve(labels3, scores3)
        fpr4, tpr4, _ = roc_curve(labels4, scores4)

        # 计算AUROC
        auroc1 = auc(fpr1, tpr1) * 100
        auroc2 = auc(fpr2, tpr2) * 100
        auroc3 = auc(fpr3, tpr3) * 100
        auroc4 = auc(fpr4, tpr4) * 100

        # 绘制AUROC曲线
        plt.figure(figsize=(5, 4), dpi=300)
        plt.grid(True)
        plt.plot(fpr1, tpr1, lw=2, label=f'EDL AUROC = {auroc1:.2f}')
        plt.plot(fpr2, tpr2, lw=2, label=f'I-EDL AUROC = {auroc2:.2f}')
        plt.plot(fpr3, tpr3, lw=2, label=f'R-EDL AUROC = {auroc3:.2f}')
        plt.plot(fpr4, tpr4, lw=2, label=f'Re-EDL AUROC = {auroc4:.2f}')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # 添加对角线
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=20, fontproperties=prop)
        plt.ylabel('True Positive Rate', fontsize=20, fontproperties=prop)
        # plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, family='Times New Roman')
        plt.legend(loc='lower right', prop=prop)

        # 设置坐标轴刻度的字体和大小
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=10)  # 设置坐标轴刻度数字的大小
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(prop)  # 设置坐标轴刻度数字的字体

        plt.tight_layout()

        # 保存高分辨率图像
        plt.savefig(f"/data_SSD1/cmy/REDL-TPAMI/code_classical/results/auroc/{uct_name}.pdf", bbox_inches='tight', dpi=250, pad_inches=0.0)

