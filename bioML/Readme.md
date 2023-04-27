1.创建环境
-
cmd执行`conda create -n xxx python==3.6`以创建环境，xxx为环境名

执行`conda activate xxx`进入环境,xxx为你的环境名

在项目目录下执行`pip install -r requirements.txt`安装依赖包

2.运行程序
-
数据集在dataset文件夹

使用`python demo3.py --data_path xxx`  以运行程序

可传入参数为：

* `--data_path` 数据路径

* `--save_name` 保存文件夹名

* `--data_type` 数据类型
  - DNA

* `--fe` 特征提取 Select feature descriptor
  - DNA_onehot

* `--cl` 聚类 Clustering
  - Kmeans

* `--cl_num` 聚类数

* `--fn` 特征标准化 Feature normalization
  - MinMax
  - Zscore
  - MaxAbs

* `--fs` 特征选择 Feature selection
  - lasso
  - Variance

* `--dr` 降维 Dimension reduction
  - PCA
  - LDA
  - Tsne

* `--dr_num` 降维数

* `--model` 回归模型 Model construction
  - LR

3.数据可视化
-
运行结果在**visualization**文件夹里

每次运行会生成一个时间戳，文件目录如下：

- bioML
  - dataset
    - dataset.txt
  - visualization 
    - timestamp 1
      - ...
    - timestamp 2
      - ...
      - plot_ROC.csv(绘制ROC的点集)
      - plot_PRC.csv(绘制PRC的点集)
      - ...
  - ...