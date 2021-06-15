#数据来源（数据集在PeMS文件）：
数据来自美国的加利福尼亚州的洛杉矶市
第一个CSV文件是关于节点的表示情况，一共有307个节点，CSV文件其实就是一个直接能可视化的邻接矩阵，当然不是我们所需的那种
from和to表示的是节点，cost表示的是两个节点之间的直线距离（来表示权重），在本实验中，权重都为1.

第二个npz文件是交通流量的文件，时间范围是两个月（2018.1.1——2018.2.28），每5分钟测一次
307个节点，每个节点三个特征，然后统计了一个月每个节点的三个特征的变换，
也就是总共有307个站点，每个站点的时序数据有16992条，每条数据有三个特征.

#数据目标：
通过交通流量数据的处理(visualize_traffic_data.py)(traffic_dataset.py)，
主要是把拿到的数据的结构信息（邻接矩阵csv）和节点信息（流量数据npz）处理成了模型所需要的train_data和test_data

#GNN模型构建：
一个好的习惯是先把整体的网络框架搭建好，之后再去实现具体的模型，这样做的好处是在后面更换模型的时候只需要改一两行代码即可。
以上使用了GCN, ChebNet, GAT三种图卷积来预测交通流量，虽然考虑到了空间的影响，但是没有考虑时序上的影响，
所以能否加入RNN模型来考虑时序影响，进一步提高预测效果？
一些模型比如：STGCN, ASTGCN, DCRNN等，都是加入了时间的影响，值得借鉴．

#无联邦学习实验结果（运行traffic_prediction文件）：
基于数据pems04：307个节点
chebnet   epoch=10
Test Loss: 0.0453
Performance:  MAE 18.32    MAPE 0.13%    RMSE 27.76

GCN epoch=10
Test Loss: 0.1395
Performance:  MAE 34.35    MAPE 0.34%    RMSE 48.44

GAT epoch=10
Test Loss: 0.1638
Performance:  MAE 36.70    MAPE 0.41%    RMSE 51.32

基于数据pems08：170个节点
ChebNet
Test Loss: 0.0362
Performance:  MAE 15.04    MAPE 0.09%    RMSE 22.41

GCN
Test Loss: 0.1863
Performance:  MAE 35.52    MAPE 0.24%    RMSE 49.69

GAT
Test Loss: 0.2418
Performance:  MAE 40.29    MAPE 0.29%    RMSE 56.19

基于数据pems07：883个节点
GCN epoch=10
Test Loss: 0.0991
Performance:  MAE 35.92    MAPE 0.17%    RMSE 53.33

chebnet   epoch=10
Test Loss: 0.0328
Performance:  MAE 20.09    MAPE 0.08%    RMSE 30.41

GAT epoch=10
Test Loss: 0.0958
Performance:  MAE 34.86    MAPE 0.19%    RMSE 51.20

# 加入联邦学习（运行所有jupyter文件）
分别是三个模型 三个数据集 两个基准模型MLP CNN




