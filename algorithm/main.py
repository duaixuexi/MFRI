import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '1':显示所有信息（默认）; '2':只显示warning与error; '3':只显示error
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4"  # 选择GPU

from algorithm import data_loader, data_process
from algorithm.model import NeuMF_J_JF_model
from algorithm.train import train

if __name__ == '__main__':
    # 初始变量
    gmf_dim, mlp_dim, layers, l2, dropout = 8, 32, [32, 16, 8], 1e-6, 0.5

    # 先读取再联合
    n_user, n_item, train_data_c, test_data_c, topk_item_data_c, train_data_s, test_data_s, topk_user_data_s = \
        data_process.pack(data_loader._read_lastfm_consum, data_loader._read_lastfm_social)
    # data_loader._read_lastfm_consum, data_loader._read_lastfm_social  # lastfm
    # data_loader._read_epinions_consum, data_loader._read_epinions_social  # Epinions 前3000评分

    model, model_c, model_s = NeuMF_J_JF_model(n_user=n_user, n_item=n_item, layers=layers, gmf_dim=gmf_dim,
                                               mlp_dim=mlp_dim, l2=l2, dropout=dropout)
    train(model, model_c, model_s, train_data_c, test_data_c, topk_item_data_c, train_data_s, test_data_s,
          topk_user_data_s, epochs=1, batch=360, lr=0.05, lr_c=0.05, lr_s=0.001)  # 1e-6、0.5、360、0.001
    # 该batch表示将数据集分为多少分（batch_size = sum_data / batch）
    # epochs=50, batch=360, lr=0.0005

    # Epinions
    # train(model, model_c, model_s, train_data_c, test_data_c, topk_item_data_c, train_data_s, test_data_s,
    #       topk_user_data_s, epochs=50, batch=4320, lr=0.005, lr_c=0.05, lr_s=0.001)  # 1e-6、0.5、360、0.001

    # lastfm
    # train(model, model_c, model_s, train_data_c, test_data_c, topk_item_data_c, train_data_s, test_data_s,
    #       topk_user_data_s, epochs=5, batch=180, lr=0.005, lr_c=0.01, lr_s=0.001)  # 1e-6、0.5、360、0.001
    #                         epochs=50, batch=360, lr=0.0005, lr_c=0.05, lr_s=0.001
