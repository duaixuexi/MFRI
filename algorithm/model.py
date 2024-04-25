'''
Wei Zhou
2024/04/25
联合神经模型
融入行为特征与互惠性的用户联合行为预测模型
MFRI
'''

import tensorflow as tf
import numpy as np
from utility.decorator import logger
from tensorflow.keras.layers import Layer

tf.random.set_seed(2022)

# 定义变量：tf.Variable
# 获取变量：tf.compat.v1.get_variable——目的是共享变量，若无变量则会自动生成新变量
# Dropout：Dense层之后加入，随机失活

@logger('初始化NeuMF_J_JF_model模型：', ('n_user', 'n_item', 'gmf_dim', 'mlp_dim', 'layers', 'l2', 'dropout'))
def NeuMF_J_JF_model(n_user: int, n_item: int, layers=[32, 16, 8], gmf_dim=8, mlp_dim=32, l2=1e-4, dropout=0.3)\
                     -> tf.keras.Model:

    # Embedding_space
    l2 = tf.keras.regularizers.l2(l2)
    # initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.05, seed=2022)
    initializer = tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.05, seed=2022)
    s_embedding = tf.keras.layers.Embedding(n_user, mlp_dim, embeddings_initializer=initializer, embeddings_regularizer=l2)
    p_embedding = tf.keras.layers.Embedding(n_user, mlp_dim, embeddings_initializer=initializer, embeddings_regularizer=l2)
    q_embedding = tf.keras.layers.Embedding(n_item, mlp_dim, embeddings_initializer=initializer, embeddings_regularizer=l2)
    x_embedding = tf.keras.layers.Embedding(n_user, gmf_dim, embeddings_initializer=initializer, embeddings_regularizer=l2)
    w_embedding = tf.keras.layers.Embedding(n_user, gmf_dim, embeddings_initializer=initializer, embeddings_regularizer=l2)
    m_embedding = tf.keras.layers.Embedding(n_item, gmf_dim, embeddings_initializer=initializer, embeddings_regularizer=l2)

    # Input
    user_id_i = tf.keras.Input(shape=(), name='user_id_i', dtype=tf.float32)
    item_id_a = tf.keras.Input(shape=(), name='item_id_a', dtype=tf.float32)
    user_id_j = tf.keras.Input(shape=(), name='user_id_j', dtype=tf.float32)

    # para_embed

    Wi = tf.keras.layers.Embedding(n_user, gmf_dim, embeddings_initializer=initializer, embeddings_regularizer=l2)(user_id_i)  # Wi
    Xi = tf.keras.layers.Embedding(n_user, gmf_dim, embeddings_initializer=initializer, embeddings_regularizer=l2)(user_id_i)  # Xi
    Pi = tf.keras.layers.Embedding(n_user, mlp_dim, embeddings_initializer=initializer, embeddings_regularizer=l2)(user_id_i)  # Pi
    Si = tf.keras.layers.Embedding(n_user, mlp_dim, embeddings_initializer=initializer, embeddings_regularizer=l2)(user_id_i)  # Si

    Ma = tf.keras.layers.Embedding(n_item, gmf_dim, embeddings_initializer=initializer, embeddings_regularizer=l2)(item_id_a)  # Ma
    Qa = tf.keras.layers.Embedding(n_item, mlp_dim, embeddings_initializer=initializer, embeddings_regularizer=l2)(item_id_a)  # Qa

    Wj = tf.keras.layers.Embedding(n_user, gmf_dim, embeddings_initializer=initializer, embeddings_regularizer=l2)(user_id_j)  # Wj
    Xj = tf.keras.layers.Embedding(n_user, gmf_dim, embeddings_initializer=initializer, embeddings_regularizer=l2)(user_id_j)  # Xj
    Pj = tf.keras.layers.Embedding(n_user, mlp_dim, embeddings_initializer=initializer, embeddings_regularizer=l2)(user_id_j)  # Pj
    Sj = tf.keras.layers.Embedding(n_user, mlp_dim, embeddings_initializer=initializer, embeddings_regularizer=l2)(user_id_j)  # Sj

    ##################################################################
    # GMF_C
    gmf_c_1 = Wi * Ma  # *：Hadamard乘积
    gmf_c_2 = Xi * Ma

    # MLP_C
    # mlp_con = tf.concat([Si, Qa, Pi, Qa], axis=1)  # 按列联结
    class MyConcatLayer(Layer):
        def call(self, inputs):
            return tf.concat(inputs, axis=1)
    mlp_con = MyConcatLayer()([Si, Qa, Pi, Qa])

    ##########################################################################################

    for n in layers:
        W = tf.compat.v1.get_variable(name='W%d' % n, shape=[mlp_con.shape[1], n], dtype=tf.float32,
                       initializer=tf.keras.initializers.glorot_uniform())
        B = tf.compat.v1.get_variable(name='B%d' % n, shape=[n], dtype=tf.float32,
                       initializer=tf.keras.initializers.glorot_uniform())
        W = tf.convert_to_tensor(W)
        B = tf.convert_to_tensor(B)

        # mlp_con = tf.matmul(mlp_con, W)

        class MatMulLayer(Layer):
            def __init__(self, **kwargs):
                super(MatMulLayer, self).__init__(**kwargs)

            def build(self, input_shape):
                self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='uniform',
                                         trainable=True)

            def call(self, inputs):
                return tf.matmul(inputs, self.W)

        mlp_con = MatMulLayer()(mlp_con)


        # mlp_con = mlp_con + B
        # mlp_con = tf.add(tf.matmul(mlp_con, W), B)
        # mlp_con = tf.keras.layers.Dropout(rate=dropout)(mlp_con)
        mlp_con = tf.keras.layers.Dense(n, activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                                        bias_initializer='zeros', kernel_regularizer=l2)(mlp_con)
        mlp_con = tf.keras.layers.Dropout(rate=dropout)(mlp_con)

    # 可行！——18+
    # mutal_atten
    ###########################################

    # gmf_con = tf.concat([gmf_c_1, gmf_c_2], axis=1)

    class ConcatenateLayer(Layer):
        def __init__(self, **kwargs):
            super(ConcatenateLayer, self).__init__(**kwargs)

        def call(self, inputs):
            return tf.concat(inputs, axis=1)

    gmf_con = ConcatenateLayer()([gmf_c_1, gmf_c_2])

    # con_hg_c = tf.concat([gmf_con, mlp_con], axis=1)
    class ConcatenateLayer(Layer):
        def __init__(self, **kwargs):
            super(ConcatenateLayer, self).__init__(**kwargs)

        def call(self, inputs):
            return tf.concat(inputs, axis=1)

    con_hg_c = ConcatenateLayer()([gmf_con, mlp_con])

    atten_gmf_c = tf.keras.layers.Dense(1, activation='softmax', kernel_regularizer=l2)(gmf_con)
    atten_gmf_c = tf.keras.layers.BatchNormalization(axis=1, trainable=True, name='atten_gmf_c')(atten_gmf_c)
    atten_mlp_c = tf.keras.layers.Dense(1, activation='softmax', kernel_regularizer=l2)(mlp_con)
    atten_mlp_c = tf.keras.layers.BatchNormalization(axis=1, trainable=True, name='atten_mlp_c')(atten_mlp_c)

    gmf_ca_c = con_hg_c * atten_mlp_c
    mlp_ca_c = con_hg_c * atten_gmf_c

    # gmf_con = tf.concat([gmf_ca_c, gmf_con], axis=1)
    class ConcatenateLayer(Layer):
        def __init__(self, **kwargs):
            super(ConcatenateLayer, self).__init__(**kwargs)

        def call(self, inputs):
            return tf.concat(inputs, axis=1)

    gmf_con = ConcatenateLayer()([gmf_ca_c, gmf_con])

    # mlp_con = tf.concat([mlp_ca_c, mlp_con], axis=1)
    class ConcatenateLayer(Layer):
        def __init__(self, **kwargs):
            super(ConcatenateLayer, self).__init__(**kwargs)

        def call(self, inputs):
            return tf.concat(inputs, axis=1)

    mlp_con = ConcatenateLayer()([mlp_ca_c, mlp_con])

    # out_con = tf.concat([mlp_con, gmf_con], axis=1)
    class ConcatenateLayer(Layer):
        def __init__(self, **kwargs):
            super(ConcatenateLayer, self).__init__(**kwargs)

        def call(self, inputs):
            return tf.concat(inputs, axis=1)

    out_con = ConcatenateLayer()([mlp_con, gmf_con])

    out_con = tf.keras.layers.Dropout(rate=dropout)(out_con)

    ###########################################

    # out_con = tf.concat([gmf_c_1, gmf_c_2, mlp_con], axis=1)
    # out_con = tf.keras.layers.Dropout(rate=dropout)(out_con)
    out_con = tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros', kernel_regularizer=l2, name='out_con')(out_con)

    ############################################################################################
    # 上为消费模型                                                                    下为社交模型 #
    ############################################################################################

    # run_social
    ##################################################################
    # GMF_S
    gmf_s_1 = Wi * Wj  # *：Hadamard乘积
    gmf_s_2 = Xi * Xj

    # gmf_s_1 = tf.multiply(Wi, Wj)  # *：Hadamard乘积
    # gmf_s_2 = tf.multiply(Xi, Xj)

    # MLP_S
    # mlp_soc = tf.concat([Si, Sj, Pi, Pj], axis=1)  # 按列联结

    class ConcatenateLayer(Layer):
        def __init__(self, **kwargs):
            super(ConcatenateLayer, self).__init__(**kwargs)

        def call(self, inputs):
            return tf.concat(inputs, axis=1)
    mlp_soc = ConcatenateLayer()([Si, Sj, Pi, Pj])

    for n in layers:
        W = tf.compat.v1.get_variable(name='W%d' % n, shape=[mlp_soc.shape[1], n], dtype=tf.float32,
                       initializer=tf.keras.initializers.glorot_uniform())
        B = tf.compat.v1.get_variable(name='B%d' % n, shape=[n], dtype=tf.float32,
                       initializer=tf.keras.initializers.glorot_uniform())
        W = tf.convert_to_tensor(W)
        B = tf.convert_to_tensor(B)
        # mlp_soc = tf.matmul(mlp_soc, W)
        class MatMulLayer(Layer):
            def __init__(self, **kwargs):
                super(MatMulLayer, self).__init__(**kwargs)

            def call(self, inputs):
                return tf.matmul(inputs[0], inputs[1])

        mlp_soc = MatMulLayer()([mlp_soc, W])

        # mlp_con = mlp_con + B
        # mlp_con = tf.add(tf.matmul(mlp_con, W), B)
        # mlp_soc = tf.keras.layers.Dropout(rate=dropout)(mlp_soc)
        mlp_soc = tf.keras.layers.Dense(n, activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                                        bias_initializer='zeros', kernel_regularizer=l2)(mlp_soc)
        mlp_soc = tf.keras.layers.Dropout(rate=dropout)(mlp_soc)

    # 可行！——18+
    # mutal_atten
    ###########################################

    # gmf_soc = tf.concat([gmf_s_1, gmf_s_2], axis=1)

    class ConcatLayer(Layer):
        def __init__(self, **kwargs):
            super(ConcatLayer, self).__init__(**kwargs)

        def call(self, inputs):
            return tf.concat(inputs, axis=1)
    gmf_soc = ConcatLayer()([gmf_s_1, gmf_s_2])

    # con_hg_s = tf.concat([gmf_soc, mlp_soc], axis=1)
    class ConcatLayer(Layer):
        def __init__(self, **kwargs):
            super(ConcatLayer, self).__init__(**kwargs)

        def call(self, inputs):
            return tf.concat(inputs, axis=1)
    con_hg_s = ConcatLayer()([gmf_soc, mlp_soc])

    atten_gmf_s = tf.keras.layers.Dense(1, activation='softmax', kernel_regularizer=l2)(gmf_soc)
    atten_gmf_s = tf.keras.layers.BatchNormalization(axis=1, trainable=True, name='atten_gmf_s')(atten_gmf_s)
    atten_mlp_s = tf.keras.layers.Dense(1, activation='softmax', kernel_regularizer=l2)(mlp_soc)
    atten_mlp_s = tf.keras.layers.BatchNormalization(axis=1, trainable=True, name='atten_mlp_s')(atten_mlp_s)

    gmf_ca_s = con_hg_s * atten_gmf_s
    mlp_ca_s = con_hg_s * atten_mlp_s

    # gmf_soc = tf.concat([gmf_ca_s, gmf_soc], axis=1)
    class ConcatLayer(Layer):
        def __init__(self, **kwargs):
            super(ConcatLayer, self).__init__(**kwargs)

        def call(self, inputs):
            return tf.concat(inputs, axis=1)
    gmf_soc = ConcatLayer()([gmf_ca_s, gmf_soc])

    # mlp_soc = tf.concat([mlp_ca_s, mlp_soc], axis=1)
    class ConcatLayer(Layer):
        def __init__(self, **kwargs):
            super(ConcatLayer, self).__init__(**kwargs)

        def call(self, inputs):
            return tf.concat(inputs, axis=1)
    mlp_soc = ConcatLayer()([mlp_ca_s, mlp_soc])

    # out_soc = tf.concat([mlp_soc, gmf_soc], axis=1)
    class ConcatLayer(Layer):
        def __init__(self, **kwargs):
            super(ConcatLayer, self).__init__(**kwargs)

        def call(self, inputs):
            return tf.concat(inputs, axis=1)
    out_soc = ConcatLayer()([mlp_soc, gmf_soc])

    out_soc = tf.keras.layers.Dropout(rate=dropout)(out_soc)

    ###########################################

    # out_soc = tf.concat([gmf_s_1, gmf_s_2, mlp_soc], axis=1)
    # out_soc = tf.keras.layers.Dropout(rate=dropout)(out_soc)
    out_soc = tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros', kernel_regularizer=l2, name='out_soc')(out_soc)

    # 结果联合
    # out_sum = tf.concat([out_con, out_soc], axis=0)
    class ConcatLayer(Layer):
        def __init__(self, **kwargs):
            super(ConcatLayer, self).__init__(**kwargs)

        def call(self, inputs):
            return tf.concat(inputs, axis=0)

    out_sum = ConcatLayer()([out_con, out_soc])

    return tf.keras.Model(inputs=[user_id_i, item_id_a, user_id_j], outputs=out_sum),\
           tf.keras.Model(inputs=[user_id_i, item_id_a], outputs=out_con),\
           tf.keras.Model(inputs=[user_id_i, user_id_j], outputs=out_soc)
