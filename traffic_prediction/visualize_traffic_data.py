
# =============================================================================
# 在处理数据之前，先看看拿到的数据长什么样子，我们可视化数据看看
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

def get_flow(file_name): # 将读取文件写成一个函数
   
    flow_data = np.load(file_name) # 载入交通流量数据
    print([key for key in flow_data.keys()]) # 打印看看key是什么  
    print(flow_data["data"].shape)  # (16992, 307, 3)，16992是时间(59*24*12)，307是节点数，3表示每一维特征的维度（类似于二维的列）
    flow_data = flow_data['data'] # [T, N, D]，T为时间，N为节点数，D为节点特征
    
    # 只取第一维特征，并且为了后面方便，将节点的维度放在第一维，所以重写的get_flow()函数如下：
    # flow_data = np.load(file_name)
    # print([key for key in flow_data.keys()])
    # print(flow_data["data"].shape)  # (16992, 307, 3)，16992是时间(59*24*12)，307是节点数，3表示每一维特征的维度（类似于二维的列）   
    # flow_data = flow_data['data'].transpose([1, 0, 2])[:, :, 0][:, :, np.newaxis]  # [N, T, D],transpose就是转置，让节点纬度在第0位，N为节点数，T为时间，D为节点特征
    # 对np.newaxis说一下，就是增加一个维度，这是因为一般特征比一个多，即使是一个，保持这样的习惯，便于通用的处理问题

    return flow_data

# 做工程、项目等第一步对拿来的数据进行可视化的直观分析
if __name__ == "__main__":

    traffic_data = get_flow("PeMS/PeMS08.npz")
    node_id = 10
    print(traffic_data.shape)
    
    plt.plot(traffic_data[:24*12, node_id, 0])  # 0维特征
    plt.savefig("asset/node_{:3d}_PEMS08_1.png".format(node_id))

    plt.plot(traffic_data[:24 * 12, node_id, 1])  # 1维特征
    plt.savefig("asset/node_{:3d}_PEMS08_2.png".format(node_id))

    plt.plot(traffic_data[:24 * 12, node_id, 2])  # 2维特征
    plt.savefig("asset/node_{:3d}_PEMS08_3.png".format(node_id))
    
    
    # 可得出：每个节点有三个特征，但是其他两个节点基本是平稳不变的，所以我们只取第一维特征。

# =============================================================================
# PEMS04输出结果：
# ['data']
# (16992, 307, 3)
# (16992, 307, 3)
# 总共有307个站点，每个站点的时序数据有16992条，每条数据有三个特征，
# PEMS08输出结果
# ['data']
# (17856, 170, 3)
# (17856, 170, 3)
# 总共有170个站点，每个站点的时序数据有17856条，每条数据有三个特征，
# =============================================================================
