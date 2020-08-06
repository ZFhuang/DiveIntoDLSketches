# coding=utf-8

# 导入自己的函数包d2lzh_pytorch，注意要先将目标包的父路径添加到系统路径中
import sys
sys.path.append(r".")
from d2lzh_pytorch import data_process

import torch
import random
import zipfile

"""
这一章要用周杰伦的歌词来进行循环神经网络生成，这一节是读取分析数据集和采样的部分
"""

# 进行数据的读取和预处理，这部分已经保存为函数load_data_jay_lyrics
# 先取得数据集压缩包
with zipfile.ZipFile(r"./Datasets/JaychouLyrics/jaychou_lyrics.txt.zip") as zin:
    # 由于是zip格式，需要解压并得到里面文件的指针
    with zin.open('jaychou_lyrics.txt') as f:
        # corpus=语料库，读取文件为数组
        corpus_chars = f.read().decode('utf-8')
# 为了处理方便将换行符替换为空格
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
# 然后仅取其前10000个字符，缩小数据集大小
corpus_chars = corpus_chars[:10000]
print(corpus_chars[:40])
print('————————————————————————————')

# 然后将数据集中的所有不同的字符映射为整数索引，方便后续处理，也就是建立索引字典
# set是不重复的字符序列，用set得到数据集不重复的部分，然后用list转为列表
idx_to_char=list(set(corpus_chars))
# 再利用上面的列表按照编号转为字符下标对应的字典
char_to_idx=dict([(char,i) for i,char in enumerate(idx_to_char)])
# 输出大小，这个大小就是前10000字中的不重复字符数量
vocab_size=len(char_to_idx)
print(vocab_size)
# 将字符替换为索引并返回
corpus_indices=[char_to_idx[char] for char in corpus_chars]
sample=corpus_indices[:20]
print('char: ',''.join([idx_to_char[idx] for idx in sample]))
print('indices:',sample)
print('————————————————————————————')

# 写一个随机采样函数data_iter_random存入自己库中待用
# 随机采样的数据训练时需要不断初始化隐藏状态，因为两个batch间无关
# 随机采样函数按照num_steps把数据集分为n个sample，然后用batch_size分成m个batch，分层
# 初始化一次data_iter_random就是从数据集中随机取出一个batch(但batch内部是连续有序的)
# 然后从开始部分取出一个sample-X和相邻一个位置的sample-Y，这两个sample是连贯的
# 接着下一个循环，会取出另一个batch开始取sample，因而两个batch间不连贯
my_seq=list(range(30))
for X,Y in data_process.data_iter_random(my_seq,batch_size=2,num_steps=6):
    print('X: ',X,'\nY:',Y,'\n')
print('————————————————————————————')

# 写一个相邻采样函数data_iter_consecutive存入自己库中待用
# 相邻采样的数据可以连续训练，因为两个batch间相邻，就像连续的sample一样
# 相邻采样的数据批与批之间是连续的，而批内的仍然有两个相邻的sample
# 这个采样的起点不是随机的，固定从0开始
my_seq=list(range(30))
for X,Y in data_process.data_iter_consecutive(my_seq,batch_size=2,num_steps=6):
    print('X: ',X,'\nY:',Y,'\n')
print('————————————————————————————')
