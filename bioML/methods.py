import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from data_visualization import eval_df, save_df, plot_cm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve


class Read_data:
    def DNAseq(self, path):
        f = open(path, 'r')
        data_txt = f.read().splitlines()
        f.close()

        data = []
        for row in range(int(len(data_txt) / 2)):
            onerow = []
            for i in range(2):
                onerow.append(data_txt[row * 2 + i])
            data.append(onerow)

        train_list = []
        test_list = []
        for i in data:
            info = i[0].split('|')
            seq = i[1]
            if info[2] == 'training':
                train_list.append([info[0].replace('>', ''), int(info[1]), seq])
            elif info[2] == 'testing':
                test_list.append([info[0].replace('>', ''), int(info[1]), seq])
            else:
                raise (f'{info[0]} mode error')

        return train_list, test_list


class Feature_extract:
    def DNA_onehot(self, lists):
        dict = {'-': np.zeros(4)}
        one_hot = np.eye(4)
        base = ['A', 'T', 'G', 'C']
        for i in range(4):
            dict[base[i]] = one_hot[i]
        rows = []
        for info in lists:
            encoder = [info[0], info[1]]
            for k in info[2]:
                encoder.extend(dict[k])
            rows.append(encoder)
        columns = ['ID', 'label']
        for j in range(len(rows[0])-2):
            columns.append(f'feature{j+1}')
        df = pd.DataFrame(rows, columns=columns)
        return df

    def NAC(self, lists):
        dic = {}
        columns = ['ID', 'label', 'A', 'T', 'G', 'C']
        for i in columns:
            dic[i] = []
        for info in lists:
            for i in range(2):
                dic[columns[i]].append(info[i])  # 'ID', 'label'
            num_list = np.zeros(4)  # A T G C
            length = len(info[2])
            for j in info[2]:
                n = 0
                for k in range(4):
                    if j == columns[k + 2]:
                        n = k
                num_list[n] += 1
            for i in range(4):
                dic[columns[i + 2]].append((round(num_list[i] / length, 6)))
        df = pd.DataFrame(dic, columns=columns)
        return df

    def ANF(self, lists):
        dic = {}
        base = ['A', 'T', 'G', 'C', '-']
        columns = ['ID', 'label']
        maxlen = max(info[2].__len__() for info in lists)

        for i in range(maxlen + 2):
            if i < maxlen:
                columns.append('ANF.' + str(i + 1))
            dic[columns[i]] = []

        for info in lists:
            for i in range(2):
                dic[columns[i]].append(info[i])  # 'ID', 'label'
            num_list = np.zeros(5)  # A T C G -
            num = 0
            for j in info[2]:
                num += 1  # 当前碱基数
                n = 0
                for k in range(5):
                    if j == base[k]:
                        n = k
                num_list[n] += 1
                dic[columns[num + 1]].append(round(num_list[n] / num, 6))
        df = pd.DataFrame(dic, columns=columns)
        return df


class Cluster:
    def Kmeans(self, train_df, test_df, n_class):
        df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
        X = df.iloc[:, 2:].reset_index(drop=True)
        cl = KMeans(n_class)
        cl.fit_transform(X)
        df_result = df.iloc[:, :1].join(pd.DataFrame(cl.labels_, columns=['Cluster']))
        return df.iloc[:, :1].join(pd.DataFrame(cl.labels_, columns=['Cluster'])).join(X), df_result


class Feature_normalization:
    def MinMax(self, df1):
        fn = MinMaxScaler()
        fn_array = fn.fit_transform(df1.iloc[:, 2:])
        columns = []
        for j in range(len(fn_array[0])):
            columns.append(f'feature{j + 1}')
        df2 = pd.DataFrame(fn_array, columns=columns)
        df = df1.iloc[:, :2].join(df2)
        return df

    def Zscore(self, df1):
        fn = StandardScaler()
        fn_array = fn.fit_transform(df1.iloc[:, 2:])
        columns = []
        for j in range(len(fn_array[0])):
            columns.append(f'feature{j + 1}')
        df2 = pd.DataFrame(fn_array, columns=columns)
        df = df1.iloc[:, :2].join(df2)
        return df

    def MaxAbs(self, df1):
        fn = MaxAbsScaler()
        fn_array = fn.fit_transform(df1.iloc[:, 2:])
        columns = []
        for j in range(len(fn_array[0])):
            columns.append(f'feature{j + 1}')
        df2 = pd.DataFrame(fn_array, columns=columns)
        df = df1.iloc[:, :2].join(df2)
        return df


class Feature_selection:
    def __init__(self, n_features):
        self.n_features = n_features

    def F_Score(self, train_df1, test_df1):
        x_train = train_df1.iloc[:, 2:].values
        x_test = test_df1.iloc[:, 2:].values
        y_train = train_df1['label'].values
        columns = train_df1.columns[2:].to_list()
        n_features = min(self.n_features, x_train.shape[1])

        # F, pvalues_f = f_classif(x_train, y_train)
        # n_features = F.shape[0] - (pvalues_f > 0.05).sum()  # 自设n_features

        fs = SelectKBest(f_classif, k=n_features)
        x_train = fs.fit_transform(x_train, y_train)
        selected_index = fs.get_support(indices=True)
        fs_ds = fs.scores_
        fs_ds[fs_ds != fs_ds] = 0
        fs_ds = round(pd.Series(fs_ds, index=columns).sort_values()[::-1], 3)
        x_test = fs.transform(x_test)

        selected_columns = []
        for i in selected_index:
            selected_columns.append(columns[i])
        col = fs_ds[:n_features].index.to_list()
        train_df2 = pd.DataFrame(x_train, columns=selected_columns)[col]
        train_df = train_df1.iloc[:, :2].join(train_df2)
        test_df2 = pd.DataFrame(x_test, columns=selected_columns)[col]
        test_df = test_df1.iloc[:, :2].join(test_df2)

        return train_df, test_df, fs_ds

    def PearsonCorrelation(self, train_df1, test_df1):
        x_train = train_df1.iloc[:, 2:].values
        x_test = test_df1.iloc[:, 2:].values
        y_train = train_df1['label'].values
        columns = train_df1.columns[2:].to_list()
        n_features = min(self.n_features, x_train.shape[1])

        fs = SelectKBest(lambda X, Y: np.array(list(map(lambda x: pearsonr(x, Y)[0], X.T))).T, k=n_features)
        x_train = fs.fit_transform(x_train, y_train)
        selected_index = fs.get_support(indices=True)
        fs_ds = fs.scores_
        fs_ds[fs_ds != fs_ds] = 0
        fs_ds = round(pd.Series(fs_ds, index=columns).sort_values()[::-1], 3)
        x_test = fs.transform(x_test)

        selected_columns = []
        for i in selected_index:
            selected_columns.append(columns[i])
        col = fs_ds[:n_features].index.to_list()
        train_df2 = pd.DataFrame(x_train, columns=selected_columns)[col]
        train_df = train_df1.iloc[:, :2].join(train_df2)
        test_df2 = pd.DataFrame(x_test, columns=selected_columns)[col]
        test_df = test_df1.iloc[:, :2].join(test_df2)

        return train_df, test_df, fs_ds

    def lasso(self, train_df1, test_df1, alpha=0.01):
        x_train = train_df1.iloc[:, 2:]
        x_test = test_df1.iloc[:, 2:]
        y_train = train_df1['label']
        fs = Lasso(alpha=alpha).fit(x_train, y_train)
        fs_ds = pd.Series(fs.coef_, index=x_train.columns)
        fs_ds = fs_ds[fs_ds != 0]
        fs_ds = fs_ds.abs().sort_values()[::-1]
        fs_array_train = np.array(x_train)[:, fs.coef_ != 0]
        fs_array_test = np.array(x_test)[:, fs.coef_ != 0]

        columns = []
        for j in range(len(fs_array_train[0])):
            columns.append(f'feature{j + 1}')
        train_df2 = pd.DataFrame(fs_array_train, columns=columns)
        train_df = train_df1.iloc[:, :2].join(train_df2)

        columns = []
        for j in range(len(fs_array_test[0])):
            columns.append(f'feature{j + 1}')
        test_df2 = pd.DataFrame(fs_array_test, columns=columns)
        test_df = test_df1.iloc[:, :2].join(test_df2)

        return train_df, test_df, fs_ds

    def variance(self, train_df1, test_df1, threshold=0.1):
        x_train = train_df1.iloc[:, 2:]
        x_test = test_df1.iloc[:, 2:]
        fs = VarianceThreshold(threshold)
        x_train = fs.fit_transform(x_train)
        x_test = np.array(x_test)[:, fs.variances_ > threshold]
        fs_ds = pd.Series(fs.variances_, index=train_df1.iloc[:, 2:].columns)
        fs_ds = fs_ds.abs().sort_values()[::-1]

        columns = []
        for j in range(len(x_train[0])):
            columns.append(f'feature{j + 1}')
        train_df2 = pd.DataFrame(x_train, columns=columns)
        train_df = train_df1.iloc[:, :2].join(train_df2)

        columns = []
        for j in range(len(x_test[0])):
            columns.append(f'feature{j + 1}')
        test_df2 = pd.DataFrame(x_test, columns=columns)
        test_df = test_df1.iloc[:, :2].join(test_df2)

        return train_df, test_df, fs_ds


class Dimension_reduction:
    def __init__(self, dimension):
        self.dimension = dimension

    def Tsne(self, df1):
        dr = TSNE(n_components=self.dimension)
        dr_array = dr.fit_transform(df1.iloc[:, 2:])
        columns = []
        for j in range(len(dr_array[0])):
            columns.append(f'feature{j + 1}')
        df2 = pd.DataFrame(dr_array, columns=columns)
        df = df1.iloc[:, :2].join(df2)
        return df

    def pca(self, df1):
        dr = PCA(n_components=self.dimension)
        dr_array = dr.fit_transform(df1.iloc[:, 2:])
        columns = []
        for j in range(len(dr_array[0])):
            columns.append(f'feature{j + 1}')
        df2 = pd.DataFrame(dr_array, columns=columns)
        df = df1.iloc[:, :2].join(df2)
        return df

    def lda(self, train_df1, test_df1):
        x_train = train_df1.iloc[:, 2:]
        y_train = train_df1['label']
        x_test = test_df1.iloc[:, 2:]
        x_min = min(np.array(x_train).min(), np.array(x_test).min())
        if x_min < 0:
            x_train -= x_min
            x_test -= x_min
        dr = LatentDirichletAllocation(n_components=self.dimension).fit(x_train, y_train)
        train_array = dr.transform(x_train)
        test_array = dr.transform(x_test)

        columns = []
        for j in range(len(train_array[0])):
            columns.append(f'feature{j + 1}')
        train_df2 = pd.DataFrame(train_array, columns=columns)
        train_df = train_df1.iloc[:, :2].join(train_df2)

        columns = []
        for j in range(len(test_array[0])):
            columns.append(f'feature{j + 1}')
        test_df2 = pd.DataFrame(test_array, columns=columns)
        test_df = test_df1.iloc[:, :2].join(test_df2)

        return train_df, test_df


class Model():
    def __init__(self, n_splits):
        self.best_model = None
        self.best_acc = 0
        self.fpr = dict()
        self.tpr = dict()
        self.precision = dict()
        self.recall = dict()
        self.statistics = []
        self.row = []
        self.L = []
        self.probs = []
        self.columns = ['Sn', 'Sp', 'Pre', 'Acc', 'MCC', 'F1', 'AUROC', 'AUPRC']
        self.n_splits = n_splits
        self.kf = KFold(n_splits=self.n_splits, shuffle=True)
        self.index = []
        for i in range(self.n_splits):
            self.index.append('Fold ' + str(i))
        self.index.extend(['Mean', 'Indep'])

    def LR(self, train_df, test_df, t):
        model = LogisticRegression(C=1.0, random_state=0)
        # model = LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1.0,
        #                            fit_intercept=True, intercept_scaling=1,
        #                            class_weight=None, random_state=1, solver='lbfgs',
        #                            max_iter=300, multi_class='auto', verbose=0,
        #                            warm_start=False, n_jobs=None, l1_ratio=None)

        Model.save_kfold(self, model, train_df, test_df, t)

    def KNN(self, train_df, test_df, t, n_neighbors=3):
        # model = KNeighborsClassifier(n_neighbors=neighbor, weights='uniform')
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform', algorithm='auto',
                                     leaf_size=30, p=2, metric='minkowski',
                                     metric_params=None, n_jobs=None)
        Model.save_kfold(self, model, train_df, test_df, t)

    def save_kfold(self, model, train_df, test_df, t):
        x_train = train_df.iloc[:, 2:].values
        y_train = train_df.iloc[:, 1].values
        x_test = test_df.iloc[:, 2:].values
        y_test = test_df.iloc[:, 1].values

        for i, (train_index, valid_index) in enumerate(self.kf.split(x_train)):
            x_train_fold, x_valid_fold = x_train[train_index], x_train[valid_index]
            y_train_fold, y_valid_fold = y_train[train_index], y_train[valid_index]
            model.fit(x_train_fold, y_train_fold)
            prediction = model.predict(x_valid_fold)
            prob = model.predict_proba(x_valid_fold)
            self.L.extend(y_valid_fold)
            self.probs.extend(prob[:, 1])
            self.row, self.fpr[i], self.tpr[i], self.precision[i], self.recall[i] = eval_df(y_valid_fold, prediction,
                                                                                            prob[:, 1], t)
            plot_df = pd.DataFrame(np.array([self.fpr[i], self.tpr[i]]).transpose(), columns=['x', 'y'])
            save_df(plot_df, f'Fold_{i}_AUROC = {self.row[6]:.4f}', t, plot_b=True, decimal=0)
            plot_df = pd.DataFrame(np.array([self.recall[i][::-1], self.precision[i][::-1]]).transpose(), columns=['x', 'y'])
            save_df(plot_df, f'Fold_{i}_AUPRC = {self.row[7]:.4f}', t, plot_b=True, decimal=0)
            self.statistics.append(self.row)

            if self.best_model == None:
                self.best_model = model
                self.best_acc = self.row[3]
            else:
                if self.row[3] > self.best_acc:
                    self.best_model = model
                    self.best_acc = self.row[3]

        mean = np.array(self.statistics).mean(axis=0).tolist()
        self.statistics.append(mean)
        self.fpr['mean'], self.tpr['mean'], _ = roc_curve(np.array(self.L), np.array(self.probs))
        self.precision['mean'], self.recall['mean'], _ = precision_recall_curve(np.array(self.L), np.array(self.probs))
        plot_df = pd.DataFrame(np.array([self.fpr['mean'], self.tpr['mean']]).transpose(), columns=['x', 'y'])
        save_df(plot_df, f'Mean_AUROC = {mean[6]:.4f}', t, plot_b=True, decimal=0)
        plot_df = pd.DataFrame(np.array([self.recall['mean'][::-1], self.precision['mean'][::-1]]).transpose(), columns=['x', 'y'])
        save_df(plot_df, f'Mean_AUPRC = {mean[7]:.4f}', t, plot_b=True, decimal=0)

        prediction = self.best_model.predict(x_test)
        prob = self.best_model.predict_proba(x_test)
        plot_cm(prediction, y_test, t)
        indep_row, self.fpr['indep'], self.tpr['indep'], self.precision['indep'], self.recall['indep'] = eval_df(y_test, prediction,
                                                                                                     prob[:, 1], t)
        plot_df = pd.DataFrame(np.array([self.fpr['indep'], self.tpr['indep']]).transpose(), columns=['x', 'y'])
        save_df(plot_df, f'Indep_AUROC = {indep_row[6]:.4f}', t, plot_b=True, decimal=0)
        plot_df = pd.DataFrame(np.array([self.recall['indep'][::-1], self.precision['indep'][::-1]]).transpose(), columns=['x', 'y'])
        save_df(plot_df, f'Indep_AUPRC = {indep_row[7]:.4f}', t, plot_b=True, decimal=0)

        self.statistics.append(indep_row)
        evals = pd.DataFrame(self.statistics, index=self.index, columns=self.columns)
        save_df(evals, 'Evaluation_metrics', t, decimal=0)
