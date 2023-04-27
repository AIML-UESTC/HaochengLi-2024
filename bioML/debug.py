import argparse
import pandas as pd

from methods import Read_data, Feature_extract, Cluster, Feature_normalization, Feature_selection, Dimension_reduction, Model
from data_visualization import eval_df, save_df, plot_cluster, plot_dr, plot_distribution


def main(args):
    print('read data')
    rd = Read_data()
    if args.data_type == 'DNA':
        train_list, test_list = rd.DNAseq(args.data_path)
    else:
        raise Exception('Not implemented')

    print('feature extract')
    fe = Feature_extract()
    if args.fe == 'onehot':
        if args.data_type == 'DNA':
            train_df = fe.DNA_onehot(train_list)
            test_df = fe.DNA_onehot(test_list)
            save_df(train_df, 'Training_set', args.save_name, decimal=2)
            save_df(test_df, 'Testing_set', args.save_name, decimal=2)
            plot_distribution(train_df, 'Training_set_distribution', args.save_name)
            plot_distribution(test_df, 'Testing_set_distribution', args.save_name)
        else:
            raise Exception('Not implemented')
    elif args.fe == 'NAC':
        train_df = fe.NAC(train_list)
        test_df = fe.NAC(test_list)
        save_df(train_df, 'Training_set', args.save_name, decimal=2)
        save_df(test_df, 'Testing_set', args.save_name, decimal=2)
        plot_distribution(train_df, 'Training_set_distribution', args.save_name)
        plot_distribution(train_df, 'Testing_set_distribution', args.save_name)
    elif args.fe == 'ANF':
        train_df = fe.ANF(train_list)
        test_df = fe.ANF(test_list)
        save_df(train_df, 'Training_set', args.save_name, decimal=2)
        save_df(test_df, 'Testing_set', args.save_name, decimal=2)
        plot_distribution(train_df, 'Training_set_distribution', args.save_name)
        plot_distribution(train_df, 'Testing_set_distribution', args.save_name)
    else:
        raise Exception('must select one feature extract method!')

    print('cluster')
    cl = Cluster()
    if args.cl == 'Kmeans':
        dr1 = Dimension_reduction(dimension=2)
        dr_train_df = dr1.Tsne(train_df)
        dr_test_df = dr1.Tsne(test_df)
        plot_df, result_df = cl.Kmeans(dr_train_df, dr_test_df, n_class=args.cl_num)
        plot_cluster(plot_df, args.cl_num, args.save_name)
        save_df(result_df, 'Clustering_result', args.save_name)
    elif args.cl != 'None':
        raise Exception('Not implemented')

    print('feature normalization')
    fn = Feature_normalization()
    if args.fn == 'MaxAbs':
        train_df = fn.MaxAbs(train_df)
        test_df = fn.MaxAbs(test_df)
        save_df(train_df, 'Normalized_training_set', args.save_name)
        save_df(test_df, 'Normalized_testing_set', args.save_name)
    elif args.fn == 'MinMax':
        train_df = fn.MinMax(train_df)
        test_df = fn.MinMax(test_df)
        save_df(train_df, 'Normalized_training_set', args.save_name)
        save_df(test_df, 'Normalized_testing_set', args.save_name)
    elif args.fn == 'Zscore':
        train_df = fn.Zscore(train_df)
        test_df = fn.Zscore(test_df)
        save_df(train_df, 'Normalized_training_set', args.save_name)
        save_df(test_df, 'Normalized_testing_set', args.save_name)
    elif args.fn != 'None':
        raise Exception('Not implemented')

    print('feature selection')
    fs = Feature_selection(args.fs_num)
    if args.fs == 'Variance':
        train_df, test_df, fs_ds = fs.variance(train_df, test_df)
        save_df(train_df, 'Selected_training_set', args.save_name)
        save_df(test_df, 'Selected_testing_set', args.save_name)
        save_df(fs_ds, 'The_top_ranked_features', args.save_name)
    elif args.fs == 'lasso':
        train_df, test_df, fs_ds = fs.lasso(train_df, test_df)
        save_df(train_df, 'Selected_training_set', args.save_name)
        save_df(test_df, 'Selected_testing_set', args.save_name)
        save_df(fs_ds, 'The_top_ranked_features', args.save_name)
    elif args.fs == 'F_Score':
        train_df, test_df, fs_ds = fs.F_Score(train_df, test_df)
        save_df(train_df, 'Selected_training_set', args.save_name)
        save_df(test_df, 'Selected_testing_set', args.save_name)
        save_df(fs_ds, 'The_top_ranked_features', args.save_name)
    elif args.fs == 'PearsonCorrelation':
        train_df, test_df, fs_ds = fs.PearsonCorrelation(train_df, test_df)
        save_df(train_df, 'Selected_training_set', args.save_name)
        save_df(test_df, 'Selected_testing_set', args.save_name)
        save_df(fs_ds, 'The_top_ranked_features', args.save_name)
    elif args.fs != 'None':
        raise Exception('Not implemented')

    print('dimension reduction')
    dr = Dimension_reduction(dimension=args.dr_num)
    if args.dr == 'PCA':
        train_df = dr.pca(train_df)
        test_df = dr.pca(test_df)
        dr_df = pd.concat([train_df, test_df])
        save_df(dr_df, 'Dimension_reduced_features', args.save_name, decimal=2)
        p_dr = Dimension_reduction(dimension=2)
        plot_train_df = p_dr.pca(train_df)
        plot_test_df = p_dr.pca(test_df)
        plot_df = pd.concat([plot_train_df, plot_test_df]).iloc[:, :4]
        plot_dr(plot_df, args.save_name)
    elif args.dr == 'LDA':
        train_df, test_df = dr.lda(train_df, test_df)
        dr_df = pd.concat([train_df, test_df])
        save_df(dr_df, 'Dimension_reduced_features', args.save_name, decimal=2)
        p_dr = Dimension_reduction(dimension=2)
        plot_train_df = p_dr.pca(train_df)
        plot_test_df = p_dr.pca(test_df)
        plot_df = pd.concat([plot_train_df, plot_test_df]).iloc[:, :4]
        plot_dr(plot_df, args.save_name)
    elif args.dr == 'Tsne':
        train_df = dr.Tsne(train_df)
        test_df = dr.Tsne(test_df)
        dr_df = pd.concat([train_df, test_df])
        save_df(dr_df, 'Dimension_reduced_features', args.save_name, decimal=2)
        p_dr = Dimension_reduction(dimension=2)
        plot_train_df = p_dr.pca(train_df)
        plot_test_df = p_dr.pca(test_df)
        plot_df = pd.concat([plot_train_df, plot_test_df]).iloc[:, :4]
        plot_dr(plot_df, args.save_name)
    elif args.dr != 'None':
        raise Exception('Not implemented')

    print('model')
    model = Model(n_splits=5)
    if args.model == 'LR':
        model.LR(train_df, test_df, args.save_name)
    elif args.model == 'KNN':
        model.KNN(train_df, test_df, args.save_name)
    elif args.model != 'None':
        raise Exception('Not implemented')


def run(dt='None', fe='None', cl='None', cl_num='None', fn='None', fs='None', fs_num='None', dr='None', dr_num='None', model='None'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'./dataset/dataset.txt')
    parser.add_argument('--save_name', type=str, default='test')
    parser.add_argument('--data_type', type=str, default='DNA')
    parser.add_argument('--fe', type=str, default='onehot')
    parser.add_argument('--cl', type=str, default='None')
    parser.add_argument('--cl_num', type=int, default=3)
    parser.add_argument('--fn', type=str, default='None')
    parser.add_argument('--fs', type=str, default='None')
    parser.add_argument('--fs_num', type=int, default=10)
    parser.add_argument('--dr', type=str, default='None')
    parser.add_argument('--dr_num', type=int, default=2)
    parser.add_argument('--model', type=str, default='KNN')
    args = parser.parse_args()
    if dt != 'None':
        args.data_type = dt
    if fe != 'None':
        args.fe = fe
    if cl != 'None':
        args.cl = cl
    if cl_num != 'None':
        args.cl_num = cl_num
    if fn != 'None':
        args.fn = fn
    if fs != 'None':
        args.fs = fs
    if fs_num != 'None':
        args.fs_num = fs_num
    if dr != 'None':
        args.dr = dr
    if dr_num != 'None':
        args.dr_num = dr_num
    if model != 'None':
        args.model = model
    main(args)


if __name__ == '__main__':
    run()
