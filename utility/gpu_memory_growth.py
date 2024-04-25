"""
import此文件后将gpu设置为显存增量模式
"""
import os

from tensorflow import config

gpus = physical_devices = config.list_physical_devices('GPU')
if len(gpus) == 0:
    print('当前没有检测到gpu，设置显存增量模式无效。')
for gpu in gpus:
    try:
        config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# def set_gpus(gpu_index):
#     if type(gpu_index) == list:
#         gpu_index = ','.join(str(_) for _ in gpu_index)
#     if type(gpu_index) == int:
#         gpu_index = str(gpu_index)
#     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
