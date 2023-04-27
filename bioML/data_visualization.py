import os

import numpy as np
import pandas as pd
import shutil
import math
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from math import sqrt
from sklearn.metrics import precision_recall_curve, confusion_matrix


def gauss(x):
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * (x ** 2))


def get_kde(x, data_array, bandwidth=0.1):
    N = len(data_array)
    res = 0
    if len(data_array) == 0:
        return 0
    for i in range(len(data_array)):
        res += gauss((x - data_array[i]) / bandwidth)
    res /= (N * bandwidth)
    return res


def plot_distribution(df, name, t):
    colors = ['b', 'r', 'y', 'g']
    array = np.array(df.iloc[:, 2:]).ravel()
    bandwidth = 1.0 * np.std(array) * (len(array) ** (-1 / 5))
    classes = list(set(df['label']))
    plt.figure()
    for l in classes:
        input_array = np.array(df[df['label'] == l].iloc[:, 2:]).ravel()
        x_array = np.linspace(input_array.min(), input_array.max(), 100)
        y_array = [get_kde(x_array[i], input_array, bandwidth) for i in range(x_array.shape[0])]
        hist_df = plt.hist(input_array, bins=9, color=colors[l], density=True)
        hist_df = pd.DataFrame(np.array([hist_df[1][:-1], hist_df[0]]).transpose(), columns=['x', 'y'])
        plt.plot(x_array.tolist(), y_array, color=colors[l], linestyle='-', label=f'Category {l}')
        plot_df = pd.DataFrame(np.array([x_array, np.array(y_array)]).transpose(), columns=['x', 'y'])
        save_df(plot_df, f'{name}_label_{l}', t, plot_b=True, decimal=0)
        save_df(hist_df, f'{name}_label_{l}_hist', t, plot_b=True, decimal=0)
    plt.legend()
    os.makedirs(rf'./visualization/{t}/png', exist_ok=True)
    plt.savefig(rf'./visualization/{t}/png/{name}.png')
    plt.close()


def plot_cluster(df, n_clusters, t):
    markers = ['x', 'o', '*', '+']
    colors = ['b', 'r', 'y', 'g']
    os.makedirs(rf'./visualization/{t}/png', exist_ok=True)
    plt.figure()
    for i in range(n_clusters):
        members = df['Cluster'] == i
        plt.scatter(df[members].iloc[:, 2], df[members].iloc[:, 3], s=60, marker=markers[i], c=colors[i], alpha=0.5)
        save_df(df[members].iloc[:, 2:4], f'Clustering_label_{i}', t, plot_b=True, decimal=0)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title(f'Clustering (n_clusters={n_clusters})')
    plt.savefig(rf'./visualization/{t}/png/Clustering.png')
    plt.close()


def plot_dr(df, t):
    classes = list(set(df['label']))
    markers = ['x', 'o', '*', '+']
    colors = ['b', 'r', 'y', 'g']
    os.makedirs(rf'./visualization/{t}/png', exist_ok=True)
    for i in classes:
        members = df['label'] == i
        plt.scatter(df[members].iloc[:, 2], df[members].iloc[:, 3], s=60,
                    marker=markers[i], c=colors[i], alpha=0.5, label=f'Category: {i}')
        save_df(df[members].iloc[:, 2:4], f'Dimension_reduction_label_{i}', t, plot_b=True, decimal=0)
    plt.legend()
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title(f'Dimension reduction')
    plt.savefig(rf'./visualization/{t}/png/Dimension_reduction.png')
    plt.close()


def save_df(df, name, t, decimal=None, plot_b=False):
    if decimal != None:
        df.iloc[:, decimal:] = df.iloc[:, decimal:].astype('float')
        df.iloc[:, decimal:] = df.iloc[:, decimal:].apply(lambda x: round(x, 4))
    if plot_b == False:
        os.makedirs(rf'./visualization/{t}/csv', exist_ok=True)
        df.to_csv(rf'./visualization/{t}/csv/{name}.csv')
    elif plot_b == True:
        os.makedirs(rf'./visualization/{t}/csv/plot', exist_ok=True)
        df.to_csv(rf'./visualization/{t}/csv/plot/{name}.csv')
    else:
        raise ValueError(f'plot_b = {plot_b} is invalid')


def eval_df(label, pred, prob, t):
    P = label.sum()
    N = len(label) - P
    TP = 0
    TN = 0
    for i in range(len(pred)):
        if pred[i] == label[i]:
            if label[i] == 1:
                TP += 1
            elif label[i] == 0:
                TN += 1
            else:
                raise('label error')
    FP = N - TN
    FN = P - TP
    Sn = TP / P
    Sp = TN / N
    Pre = TP / (TP + FP)
    Acc = (TP + TN) / (P + N)
    MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    fpr, tpr, thresholds = roc_curve(label, prob)
    auroc = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(label, prob)
    auprc = auc(recall, precision)
    F1_score = 2 * Pre * Sn / (Pre + Sn)
    return [Sn, Sp, Pre, Acc, MCC, F1_score, auroc, auprc], fpr, tpr, precision, recall


def plot_cm(pred, label, t):
    classes = set(label)
    x = np.zeros(len(classes) ** 2)
    y = np.zeros(len(classes) ** 2)
    z = np.zeros(len(classes) ** 2)
    cm = confusion_matrix(pred, label)
    k = 0
    for i in range(len(classes)):
        for j in range(len(classes)):
            x[k] = i
            y[k] = j
            z[k] = cm[i, j]
            k += 1
    save_df(pd.DataFrame(np.array([x, y, z]).transpose(), columns=['x', 'y', 'z']), 'Confusion_matrix', t, plot_b=True)
    classes = list(set(label))
    os.makedirs(rf'./visualization/{t}/png', exist_ok=True)
    plt.figure()
    plt.imshow(cm, cmap=plt.cm.Blues)
    indices = range(len(cm))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('pred')
    plt.ylabel('true')
    for first_index in range(len(cm)):
        for second_index in range(len(cm[first_index])):
            plt.text(first_index, second_index, cm[first_index][second_index])
    plt.savefig(rf'./visualization/{t}/png/Confusion_matrix.png')
    plt.close()


def plot_ROC(label, prob, t):
    fpr, tpr, thresholds = roc_curve(label, prob)
    auroc = auc(fpr, tpr)
    os.makedirs(rf'./visualization/{t}/png', exist_ok=True)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Indep (area = {:.3f})'.format(auroc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(rf'./visualization/{t}/png/ROC.png')
    plt.close()
    return auroc, fpr, tpr


def plot_PRC(label, prob, t):
    precision, recall, thresholds = precision_recall_curve(label, prob)
    auprc = auc(recall, precision)
    os.makedirs(rf'./visualization/{t}/png', exist_ok=True)
    plot_df = pd.DataFrame(np.array([recall, precision]).transpose(), columns=['x', 'y'])
    save_df(plot_df, 'PRC', t, plot_b=True, decimal=1)
    plt.figure()
    plt.plot(recall, precision, label='Indep (area = {:.3f})'.format(auprc))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PRC curve')
    plt.legend(loc='best')
    plt.savefig(rf'./visualization/{t}/png/PRC.png')
    plt.close()
    return auprc


def init_visualization(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


if __name__ == '__main__':
    init_visualization(r'./visualization')
