import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from methods import Read_data, Feature_extract
from debug import run

dataPath = r'./dataset/dataset.txt'


def read_data():
    f = open(dataPath, 'r')
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
            train_list.append(info[0].replace('>', '') + ' ' + str(info[1]) + ' ' + seq)
        elif info[2] == 'testing':
            test_list.append(info[0].replace('>', '') + ' ' + str(info[1]) + ' ' + seq)
        else:
            raise (f'{info[0]} mode error')

    return train_list, test_list


def main():
    st.title("合成生物机器学习")

    selection = st.sidebar.selectbox("选项", ["数据展示", "分类展示"])

    if selection == "数据展示":
        st.subheader("数据展示")
        if st.checkbox('Show raw data'):
            trainlist, testlist = read_data()
            st.markdown('训练数据')
            st.write(trainlist)
            st.markdown('测试数据')
            st.write(testlist)

    if selection == "分类展示":
        st.sidebar.subheader("选择参数")
        dt = st.sidebar.selectbox('数据类型', ['DNA'])
        fe = st.sidebar.selectbox('特征提取方式', ['onehot', 'NAC', 'ANF'])

        if dt == 'DNA':
            train_list, test_list = Read_data().DNAseq(dataPath)
        else:
            raise Exception('Not implemented')

        FE = Feature_extract()
        if fe == 'onehot':
            if dt == 'DNA':
                train_df = FE.DNA_onehot(train_list)
            else:
                raise Exception('Not implemented')
        elif fe == 'NAC':
            train_df = FE.NAC(train_list)
        elif fe == 'ANF':
            train_df = FE.ANF(train_list)
        else:
            raise Exception('must select one feature extract method!')

        cl = st.sidebar.selectbox('聚类', ['Kmeans', 'None'])
        if cl != 'None':
            cl_num = st.sidebar.radio('聚类数', [2, 3, 4])
        else:
            cl_num = 'None'
        fn = st.sidebar.selectbox('标准化', ['MaxAbs', 'MinMax', 'Zscore', 'None'])
        fs = st.sidebar.selectbox('特征筛选', ['Variance', 'lasso', 'F_Score', 'PearsonCorrelation', 'None'])
        if fs != 'None':
            fnum = len(train_df.columns) - 2
            fs_num = st.sidebar.slider('特征筛选数', 1, fnum, int(fnum/2))
        else:
            fs_num = 'None'
        dr = st.sidebar.selectbox('降维', ['PCA', 'LDA', 'Tsne', 'None'])
        if dr != 'None':
            dr_num = st.sidebar.slider('特征筛选数', 1, 5, 2)
        else:
            dr_num = 'None'
        model = st.sidebar.selectbox('模型', ['LR', 'KNN', 'None'])

        start = st.sidebar.button('开始分析')
        if start:
            run(dt, fe, cl, cl_num, fn, fs, fs_num, dr, dr_num, model)
            root = r'./visualization/test'

            st.subheader("结果展示")

            st.markdown('数据分布')
            train_dis = np.array(Image.open(root + '/png/Training_set_distribution.png'), dtype=np.uint8)
            test_dis = np.array(Image.open(root + '/png/Testing_set_distribution.png'), dtype=np.uint8)
            st.image([train_dis, test_dis], ['训练数据', '测试数据'])

            if cl != 'None':
                st.markdown('聚类散点图')
                dr_png = np.array(Image.open(root + '/png/Clustering.png'), dtype=np.uint8)
                st.image(dr_png)

            if dr != 'None':
                st.markdown('降维散点图')
                dr_png = np.array(Image.open(root + '/png/Dimension_reduction.png'), dtype=np.uint8)
                st.image(dr_png)

            st.markdown('混淆矩阵')
            cm = np.array(Image.open(root + '/png/Confusion_matrix.png'), dtype=np.uint8)
            st.image(cm)

            # st.markdown('ROC曲线')
            # roc_pd_f0 = pd.read_csv(root + '/csv/plot/Fold_0_ROC.csv')
            # roc_pd_f1 = pd.read_csv(root + '/csv/plot/Fold_1_ROC.csv')
            # roc_pd_f2 = pd.read_csv(root + '/csv/plot/Fold_2_ROC.csv')
            # roc_pd_f3 = pd.read_csv(root + '/csv/plot/Fold_3_ROC.csv')
            # roc_pd_f4 = pd.read_csv(root + '/csv/plot/Fold_4_ROC.csv')
            # roc_pd_indep = pd.read_csv(root + '/csv/plot/Indep_ROC.csv')
            # roc_pd_mean = pd.read_csv(root + '/csv/plot/Mean_ROC.csv')
            # prc_pd_f0 = pd.read_csv(root + '/csv/plot/Fold_0_PRC.csv')
            # prc_pd_f1 = pd.read_csv(root + '/csv/plot/Fold_1_PRC.csv')
            # prc_pd_f2 = pd.read_csv(root + '/csv/plot/Fold_2_PRC.csv')
            # prc_pd_f3 = pd.read_csv(root + '/csv/plot/Fold_3_PRC.csv')
            # prc_pd_f4 = pd.read_csv(root + '/csv/plot/Fold_4_PRC.csv')
            # prc_pd_indep = pd.read_csv(root + '/csv/plot/Indep_PRC.csv')
            # prc_pd_mean = pd.read_csv(root + '/csv/plot/Mean_PRC.csv')
            # st.line_chart(pd.concat([roc_pd_f0, roc_pd_f1, roc_pd_f2, roc_pd_f3, roc_pd_f4, roc_pd_mean, roc_pd_indep]))

            st.markdown('指标')
            df = pd.read_csv(root + '/csv/Evaluation_metrics.csv')
            st.dataframe(df)


if __name__ == '__main__':
    main()
