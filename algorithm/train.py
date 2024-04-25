import time
from typing import List, Tuple
import tensorflow as tf
import numpy as np
from algorithm.common import log, topk
from algorithm.evaluation import TopkData
from utility.decorator import logger

#  训练集&测试集按照batch进行切分，先后训练
def prepare_ds_c(train_data: List[Tuple[int, int, int]], test_data: List[Tuple[int, int, int]],
                 batch: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    def xy(data):
        user_ids = tf.constant([d[0] for d in data], dtype=tf.int32)
        item_ids = tf.constant([d[1] for d in data], dtype=tf.int32)
        labels = tf.constant([d[2] for d in data], dtype=tf.keras.backend.floatx())
        return {'user_id_i': user_ids, 'item_id_a': item_ids}, labels

    train_ds = tf.data.Dataset.from_tensor_slices(xy(train_data)).shuffle(len(train_data)).batch(np.floor(len(train_data) / batch))
    test_ds = tf.data.Dataset.from_tensor_slices(xy(test_data)).batch(np.floor(len(test_data) / batch))

    return train_ds, test_ds

def prepare_ds_s(train_data: List[Tuple[int, int, int]], test_data: List[Tuple[int, int, int]],
                 batch: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    def xy(data):
        user_ids = tf.constant([d[0] for d in data], dtype=tf.int32)
        item_ids = tf.constant([d[1] for d in data], dtype=tf.int32)
        labels = tf.constant([d[2] for d in data], dtype=tf.keras.backend.floatx())
        return {'user_id_i': user_ids, 'user_id_j': item_ids}, labels

    train_ds = tf.data.Dataset.from_tensor_slices(xy(train_data)).shuffle(len(train_data)).batch(np.floor(len(train_data) / batch))
    test_ds = tf.data.Dataset.from_tensor_slices(xy(test_data)).batch(np.floor(len(test_data) / batch))

    return train_ds, test_ds

######################################################################################

#  计算各评价指标
def _evaluate(model, dataset, loss_object, mean_metric=tf.keras.metrics.Mean(), auc_metric=tf.keras.metrics.AUC(),
              precision_metric=tf.keras.metrics.Precision(), recall_metric=tf.keras.metrics.Recall()):

    @tf.function
    def evaluate_batch(ui, label):
        score = tf.squeeze(model(ui))  # 从张量形状中移除大小为1的维度
        loss = loss_object(label, score) + sum(model.losses)
        return score, loss

    for ui, label in dataset:
        score, loss = evaluate_batch(ui, label)

        mean_metric.update_state(loss)
        auc_metric.update_state(label, score)
        precision_metric.update_state(label, score)
        recall_metric.update_state(label, score)

    return mean_metric.result(), auc_metric.result(), precision_metric.result(), recall_metric.result()

######################################################################################

# 各epoch得到的精度进行保存，最终输出最优的精度
precision_c_list, recall_c_list, f1_c_list, ndcg_c_list, precision_s_list, recall_s_list, f1_s_list, ndcg_s_list = [], [], [], [], [], [], [], []

#  模型训练：graph/eager
#  graph执行模式
def _train_graph(model, model_c, model_s, train_ds_c, test_ds_c, topk_item_data_c, train_ds_s, test_ds_s,
                 topk_user_data_s, optimizer, optimizer_c, optimizer_s, loss_object_c, loss_object_s, epochs):
    score_fn = get_score_fn(model)
    score_fn_c = get_score_fn(model_c)
    score_fn_s = get_score_fn(model_s)


    @tf.function
    def train_batch(ui_c, label_c, ui_s, label_s):
        with tf.GradientTape(persistent=True) as tape:
            score_c = tf.squeeze(model_c(ui_c, training=True))
            score_s = tf.squeeze(model_s(ui_s, training=True))
            loss_c = loss_object_c(label_c, score_c) + sum(model_c.losses)
            loss_s = loss_object_s(label_s, score_s) + sum(model_s.losses)
            loss = 0.2 * loss_object_c(label_c, score_c) + 0.4 * loss_object_s(label_s, score_s) + sum(model.losses)  # 训练损失（真值：label、预测：score）+正则化损失
        # 社交模型训练
        # gradients_s = tape.gradient(loss_s, model_s.trainable_variables)
        # optimizer_s.apply_gradients(zip(gradients_s, model_s.trainable_variables))

        # 消费模型训练
        # gradients_c = tape.gradient(loss_c, model_c.trainable_variables)
        # optimizer_c.apply_gradients(zip(gradients_c, model_c.trainable_variables))

        # 联合模型训练
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


    for epoch in range(epochs):
        # 优化模型
        time1 = time.time()

        ui_c_list, label_c_list, ui_s_list, label_s_list = [], [], [], []
        for ui_c, label_c in train_ds_c:
            ui_c_list.append(ui_c)
            label_c_list.append(label_c)
        for ui_s, label_s in train_ds_s:
            ui_s_list.append(ui_s)
            label_s_list.append(label_s)
        for ui_c, label_c, ui_s, label_s in zip(ui_c_list, label_c_list, ui_s_list, label_s_list):
            train_batch(ui_c, label_c, ui_s, label_s)
        time2 = time.time()
        print('第%d次优化模型耗费%d秒' % (epoch + 1, (time2 - time1)))

        # 计算误差
        train_loss_c, train_auc_c, train_precision_c, train_recall_c = _evaluate(model_c, train_ds_c, loss_object_c)
        train_loss_s, train_auc_s, train_precision_s, train_recall_s = _evaluate(model_s, train_ds_s, loss_object_s)
        test_loss_c, test_auc_c, test_precision_c, test_recall_c = _evaluate(model_c, test_ds_c, loss_object_c)
        test_loss_s, test_auc_s, test_precision_s, test_recall_s = _evaluate(model_s, test_ds_s, loss_object_s)

        log(epoch, train_loss_c, train_auc_c, train_precision_c, train_recall_c,
            test_loss_c, test_auc_c, test_precision_c, test_recall_c)
        log(epoch, train_loss_s, train_auc_s, train_precision_s, train_recall_s,
            test_loss_s, test_auc_s, test_precision_s, test_recall_s)
        time3 = time.time()
        print('第%d次计算误差耗费%d秒' % (epoch + 1, (time3 - time2)))

        # topk(topk_item_data_c, topk_user_data_s, score_fn, score_fn_c, score_fn_s)
        precision_c, recall_c, f1_c, ndcg_c, precision_s, recall_s, f1_s, ndcg_s\
            = topk(topk_item_data_c, topk_user_data_s, score_fn, score_fn_c, score_fn_s)
        precision_c_list.append(precision_c)
        recall_c_list.append(recall_c)
        f1_c_list.append(f1_c)
        ndcg_c_list.append(ndcg_c)
        precision_s_list.append(precision_s)
        recall_s_list.append(recall_s)
        f1_s_list.append(f1_s)
        ndcg_s_list.append(ndcg_s)
        time4 = time.time()
        print('第%d次计算精度耗费%d秒' % (epoch + 1, (time4 - time3)))
        print(' ')

    # 输出最大值
    print('-----------------------------------分割线-----------------------------------')
    print('消费模型中各精度最大值：precision=%.4f, recall=%.4f, f1=%.4f, NDCG=%.4f'
          % (max(precision_c_list)[0], max(recall_c_list)[0], max(f1_c_list), max(ndcg_c_list)[0]))
    print('社交模型中各精度最大值：precision=%.4f, recall=%.4f, f1=%.4f, NDCG=%.4f'
          % (max(precision_s_list)[0], max(recall_s_list)[0], max(f1_s_list), max(ndcg_s_list)[0]))

######################################################################################

#  eager执行模式
# def _train_eager(model, model_c, model_s, train_ds_c, test_ds_c, topk_item_data_c,
#                  train_ds_s, test_ds_s, topk_user_data_s, optimizer, loss_object, epochs):
#     score_fn = get_score_fn(model)
#     score_fn_c = get_score_fn(model_c)
#     score_fn_s = get_score_fn(model_s)
#     # model.compile(optimizer=optimizer, loss=loss_object, metrics=['AUC', 'Precision', 'Recall'])
#     # model.fit(train_ds, epochs=epochs, verbose=1, validation_data=test_ds,
#     #           callbacks=[EarlyStopping(monitor='val_loss', patience=2), RsCallback(topk_data, get_score_fn(model))])
#     model_c.compile(optimizer=optimizer, loss=loss_object, metrics=['AUC', 'Precision', 'Recall'])
#     model_c.fit(train_ds_c, epochs=epochs, verbose=1, validation_data=test_ds_c,
#                 callbacks=[EarlyStopping(monitor='precision', patience=2, mode='max'),
#                            RsCallback(topk_item_data_c, score_fn_c)])
#
#     model_s.compile(optimizer=optimizer, loss=loss_object, metrics=['AUC', 'Precision', 'Recall'])
#     model_s.fit(train_ds_c, epochs=epochs, verbose=1, validation_data=test_ds_c,
#                 callbacks=[EarlyStopping(monitor='precision', patience=2, mode='max'),
#                            RsCallback(topk_item_data_c, score_fn_s)])
#
# #  回调函数
# class RsCallback(tf.keras.callbacks.Callback):
#     def __init__(self, topk_data: TopkData, score_fn: Callable[[Dict[str, List[int]]], List[float]]):
#         super(RsCallback, self).__init__()
#         self.topk_data = topk_data
#         self.score_fn = score_fn
#
#
#     def on_epoch_end(self, epoch, logs=None):
#         log(epoch, logs['loss'], logs['auc'], logs['precision'], logs['recall'],
#             logs['val_loss'], logs['val_auc'], logs['val_precision'], logs['val_recall'])
#
#         topk(self.topk_data, self.score_fn)
#         topk(topk_item_data_c, topk_user_data_s, score_fn, score_fn_c, score_fn_s)

######################################################################################
#  通过训练集迭代模型
@logger('开始训练，', ('epochs', 'batch'))
def train(model: tf.keras.Model, model_c: tf.keras.Model, model_s: tf.keras.Model,
          train_data_c: List[Tuple[int, int, int]], test_data_c: List[Tuple[int, int, int]], topk_item_data_c: TopkData,
          train_data_s: List[Tuple[int, int, int]], test_data_s: List[Tuple[int, int, int]], topk_user_data_s: TopkData,
          epochs=100, batch=512, lr=0.001, lr_c=0.005, lr_s=0.0005) -> None:
    """
    通用训练流程。

    :param model: 模型
    :param train_data: 训练集
    :param test_data: 测试集
    :param topk_data: 用于topk评估数据
    :param optimizer: 优化器，默认为Adam
    :param loss_object: 损失函数，默认为BinaryCrossentropy
    :param epochs: 迭代次数
    :param batch: 批数量
    :param execution: 执行模式，为eager或graph。在eager模式下，用model.fit；在graph模式下，用tf.function和GradientTape
    """

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    optimizer_c = tf.keras.optimizers.Adam(learning_rate=lr_c)
    optimizer_s = tf.keras.optimizers.SGD(learning_rate=lr_s)

    loss_object_c = tf.keras.losses.BinaryCrossentropy()
    loss_object_s = tf.keras.losses.MeanSquaredError()  # MeanSquaredError()

    train_ds_c, test_ds_c = prepare_ds_c(train_data_c, test_data_c, batch)
    train_ds_s, test_ds_s = prepare_ds_s(train_data_s, test_data_s, batch)

    _train_graph(model, model_c, model_s, train_ds_c, test_ds_c, topk_item_data_c,
                 train_ds_s, test_ds_s, topk_user_data_s, optimizer, optimizer_c,
                 optimizer_s, loss_object_c, loss_object_s, epochs)


def get_score_fn(model):
    @tf.function(experimental_relax_shapes=True)  # 放宽参数形状，从而避免不必要的跟踪
    def _fast_model(ui):
        return tf.squeeze(model(ui))  # 从张量形状中移除大小为1的维度

    def score_fn(ui):
        ui = {k: tf.constant(v, dtype=tf.int32) for k, v in ui.items()}  # k、v分别是键、值
        return _fast_model(ui).numpy()

    return score_fn
