import sys
import pandas as pd

from methods import Read_data, Feature_extract, Cluster, Feature_normalization, Feature_selection, Dimension_reduction, Model
from data_visualization import eval_df, save_df, plot_cluster, plot_dr, plot_distribution


if __name__ == '__main__':

    dataPath = sys.argv[1]
    saveName = sys.argv[2]
    dataType = sys.argv[3]
    fe = sys.argv[4]
    cl = sys.argv[5]
    if sys.argv[6] == 'None':
        clNum = 0
    else:
        clNum = int(sys.argv[6])
    fn = sys.argv[7]
    fs = sys.argv[8]
    if sys.argv[9] == 'None':
        fsNum = 0
    else:
        fsNum = int(sys.argv[9])
    dr = sys.argv[10]
    if sys.argv[11] == 'None':
        drNum = 0
    else:
        drNum = int(sys.argv[11])
    model = sys.argv[12]

    dict = {'data_path': dataPath, 'save_name': saveName, 'data_type': dataType, 'fe': fe, 'cl': cl, 'cl_num': clNum,
            'fn': fn, 'fs': fs, 'fs_num': fsNum, 'dr': dr, 'dr_num': drNum, 'model': model}

    rd = Read_data()
    if dict.get('data_type') == 'DNA':
        train_list, test_list = rd.DNAseq(dict.get('data_path'))
    else:
        raise Exception('Not implemented')

    fe = Feature_extract()
    if dict.get('fe') == 'binary':
        if dict.get('data_type') == 'DNA':
            train_df = fe.DNA_onehot(train_list)
            test_df = fe.DNA_onehot(test_list)
            save_df(train_df, 'Training_set', dict.get('save_name'), decimal=2)
            save_df(test_df, 'Testing_set', dict.get('save_name'), decimal=2)
            plot_distribution(train_df, 'Training_set_distribution', dict.get('save_name'))
            plot_distribution(test_df, 'Testing_set_distribution', dict.get('save_name'))
        else:
            raise Exception('Not implemented')
    elif dict.get('fe') == 'NAC':
        train_df = fe.NAC(train_list)
        test_df = fe.NAC(test_list)
        save_df(train_df, 'Training_set', dict.get('save_name'), decimal=2)
        save_df(test_df, 'Testing_set', dict.get('save_name'), decimal=2)
        plot_distribution(train_df, 'Training_set_distribution', dict.get('save_name'))
        plot_distribution(train_df, 'Testing_set_distribution', dict.get('save_name'))
    elif dict.get('fe') == 'ANF':
        train_df = fe.ANF(train_list)
        test_df = fe.ANF(test_list)
        save_df(train_df, 'Training_set', dict.get('save_name'), decimal=2)
        save_df(test_df, 'Testing_set', dict.get('save_name'), decimal=2)
        plot_distribution(train_df, 'Training_set_distribution', dict.get('save_name'))
        plot_distribution(train_df, 'Testing_set_distribution', dict.get('save_name'))
    else:
        raise Exception('must select one feature extract method!')

    cl = Cluster()
    if dict.get('cl') == 'Kmeans':
        dr1 = Dimension_reduction(dimension=2)
        dr_train_df = dr1.Tsne(train_df)
        dr_test_df = dr1.Tsne(test_df)
        plot_df, result_df = cl.Kmeans(dr_train_df, dr_test_df, n_class=dict.get('cl_num'))
        plot_df.rename(columns={'feature1': 'PC 1', 'feature2': 'PC 2'})
        plot_cluster(plot_df, dict.get('cl_num'), dict.get('save_name'))
        save_df(result_df, 'Clustering_result', dict.get('save_name'))
    elif dict.get('cl') != 'None':
        raise Exception('Not implemented')

    fn = Feature_normalization()
    if dict.get('fn') == 'MaxAbs':
        train_df = fn.MaxAbs(train_df)
        test_df = fn.MaxAbs(test_df)
        save_df(train_df, 'Normalized_training_set', dict.get('save_name'))
        save_df(test_df, 'Normalized_testing_set', dict.get('save_name'))
    elif dict.get('fn') == 'MinMax':
        train_df = fn.MinMax(train_df)
        test_df = fn.MinMax(test_df)
        save_df(train_df, 'Normalized_training_set', dict.get('save_name'))
        save_df(test_df, 'Normalized_testing_set', dict.get('save_name'))
    elif dict.get('fn') == 'Zscore':
        train_df = fn.Zscore(train_df)
        test_df = fn.Zscore(test_df)
        save_df(train_df, 'Normalized_training_set', dict.get('save_name'))
        save_df(test_df, 'Normalized_testing_set', dict.get('save_name'))
    elif dict.get('fn') != 'None':
        raise Exception('Not implemented')

    fs = Feature_selection(dict.get('fs_num'))
    if dict.get('fs') == 'Variance':
        train_df, test_df, fs_ds = fs.variance(train_df, test_df)
        save_df(train_df, 'Selected_training_set', dict.get('save_name'))
        save_df(test_df, 'Selected_testing_set', dict.get('save_name'))
        save_df(fs_ds, 'The_top_ranked_features', dict.get('save_name'))
    elif dict.get('fs') == 'lasso':
        train_df, test_df, fs_ds = fs.lasso(train_df, test_df)
        save_df(train_df, 'Selected_training_set', dict.get('save_name'))
        save_df(test_df, 'Selected_testing_set', dict.get('save_name'))
        save_df(fs_ds, 'The_top_ranked_features', dict.get('save_name'))
    elif dict.get('fs') == 'F_Score':
        train_df, test_df, fs_ds = fs.F_Score(train_df, test_df)
        save_df(train_df, 'Selected_training_set', dict.get('save_name'))
        save_df(test_df, 'Selected_testing_set', dict.get('save_name'))
        save_df(fs_ds, 'The_top_ranked_features', dict.get('save_name'))
    elif dict.get('fs') == 'PearsonCorrelation':
        train_df, test_df, fs_ds = fs.PearsonCorrelation(train_df, test_df)
        save_df(train_df, 'Selected_training_set', dict.get('save_name'))
        save_df(test_df, 'Selected_testing_set', dict.get('save_name'))
        save_df(fs_ds, 'The_top_ranked_features', dict.get('save_name'))
    elif dict.get('fs') != 'None':
        raise Exception('Not implemented')

    dr = Dimension_reduction(dimension=dict.get('dr_num'))
    if dict.get('dr') == 'PCA':
        train_df = dr.pca(train_df)
        test_df = dr.pca(test_df)
        dr_df = pd.concat([train_df, test_df])
        save_df(dr_df, 'Dimension_reduced_features', dict.get('save_name'), decimal=2)
        p_dr = Dimension_reduction(dimension=2)
        plot_train_df = p_dr.pca(train_df)
        plot_test_df = p_dr.pca(test_df)
        plot_df = pd.concat([plot_train_df, plot_test_df]).iloc[:, :4]
        plot_dr(plot_df, dict.get('save_name'))
    elif dict.get('dr') == 'LDA':
        train_df, test_df = dr.lda(train_df, test_df)
        dr_df = pd.concat([train_df, test_df])
        save_df(dr_df, 'Dimension_reduced_features', dict.get('save_name'), decimal=2)
        p_dr = Dimension_reduction(dimension=2)
        plot_train_df = p_dr.pca(train_df)
        plot_test_df = p_dr.pca(test_df)
        plot_df = pd.concat([plot_train_df, plot_test_df]).iloc[:, :4]
        plot_dr(plot_df, dict.get('save_name'))
    elif dict.get('dr') == 'Tsne':
        train_df = dr.Tsne(train_df)
        test_df = dr.Tsne(test_df)
        dr_df = pd.concat([train_df, test_df])
        save_df(dr_df, 'Dimension_reduced_features', dict.get('save_name'), decimal=2)
        p_dr = Dimension_reduction(dimension=2)
        plot_train_df = p_dr.pca(train_df)
        plot_test_df = p_dr.pca(test_df)
        plot_df = pd.concat([plot_train_df, plot_test_df]).iloc[:, :4]
        plot_dr(plot_df, dict.get('save_name'))
    elif dict.get('dr') != 'None':
        raise Exception('Not implemented')

    model = Model(n_splits=5)
    if dict.get('model') == 'LR':
        model.LR(train_df, test_df, dict.get('save_name'))
    elif dict.get('model') == 'KNN':
        model.KNN(train_df, test_df, dict.get('save_name'))
    elif dict.get('model') != 'None':
        raise Exception('Not implemented')