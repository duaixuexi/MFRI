import numpy as np
import tensorflow.keras as keras

np.random.seed(1)
print(np.random.random())
print(np.random.random())


def mapk(actual, predicted, k):
    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def precisionk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(predicted)


def recallk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(actual)


def ndcgk(actual, predicted, k):
    idcg = 1.0
    dcg = 1.0 if predicted[0] in actual else 0.0
    for i,p in enumerate(predicted[1:]):
        if p in actual:
            dcg += 1.0 / np.log(i+2)
        idcg += 1.0 / np.log(i+2)
    return dcg / idcg


#########################可用初始化器汇总########################
keras.initializers.Initializer()  # 初始化器基类：所有初始化器继承这个类。
keras.initializers.Zeros()  # 将张量初始值设为 0 的初始化器。
keras.initializers.Ones()  # 将张量初始值设为 1 的初始化器。将张量初始值设为 1 的初始化器。将张量初始值设为 1 的初始化器。
keras.initializers.Constant(value=0)  # 将张量初始值设为一个常数的初始化器。value: 浮点数，生成的张量的值。
keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)  # 按照正态分布生成随机张量的初始化器。
keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)  # 按照均匀分布生成随机张量的初始化器。
keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)  # 按照截尾正态分布生成随机张量的初始化器。
keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)  # 初始化器能够根据权值的尺寸调整其规模。
keras.initializers.Orthogonal(gain=1.0, seed=None)  # 生成一个随机正交矩阵的初始化器。
keras.initializers.Identity(gain=1.0)  # 生成单位矩阵的初始化器。
keras.initializers.lecun_uniform(seed=None)  # LeCun 均匀初始化器。
keras.initializers.glorot_normal(seed=None)  # keras.initializers.glorot_normal(seed=None)
keras.initializers.glorot_uniform(seed=None)  # Glorot 均匀分布初始化器，也称为 Xavier 均匀分布初始化器。
keras.initializers.he_normal(seed=None)  # He 正态分布初始化器。
keras.initializers.lecun_normal(seed=None)  # LeCun 正态分布初始化器。
keras.initializers.he_uniform(seed=None)  # He 均匀方差缩放初始化器。

