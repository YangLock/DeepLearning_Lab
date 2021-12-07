import tensorflow as tf
import tensorflow.keras as tfkeras
import tensorflow_datasets as tfds
import pandas as pd
from sklearn.utils import shuffle

EN_FILENAME_PREFIX = 'en_vocab'
CN_FILENAME_PREFIX = 'cn_vocab'

TRAIN_PATH = './NMT_en2zh/data/train.txt'
TEST_PATH = './NMT_en2zh/data/test.txt'

def split_train_test(origin_path, train_path, test_path):
    '''
    将原数据集打乱，并按照7:3的比例拆分成训练数据集和测试数据集，然后分别将训练数据和测试数据写入两个新文件中。

    Params:
    ------
        origin_path: 原始数据文件的存储路径
        train_path: 训练数据集的存储路径
        test_path: 测试数据集的存储路径
    '''
    # 获取初始数据文件加载进来的原始数据，格式为（英文 \t 中文 \t 描述）
    meta_data = pd.read_table(origin_path)
    meta_data.columns = ['English', 'Chinese', 'Description']

    # 从原始数据中截取出前两列的数据作为本次实验要用到的数据
    data_for_lab = meta_data.iloc[:, 0:2]
    # 将数据打乱
    data_for_lab = shuffle(data_for_lab)

    # 按照7:3的比例将数据分割成训练集和测试集
    total_data_size = data_for_lab.shape[0]
    train_data_size = (int)(total_data_size * 0.7)
    train_examples = data_for_lab.iloc[0:train_data_size, :]
    test_examples = data_for_lab.iloc[train_data_size:total_data_size, :]

    # 写入训练数据集
    with open(file=train_path, mode='w', encoding='utf-8') as f:
        for en, cn in zip(train_examples['English'], train_examples['Chinese']):
            f.write(en + '\t' + cn)
            f.write('\n')

    # 写入测试数据集
    with open(file=test_path, mode='w', encoding='utf-8') as f:
        for en, cn in zip(test_examples['English'], test_examples['Chinese']):
            f.write(en + '\t' + cn)
            f.write('\n')

def get_data(file_path):
    '''
    从指定文件中读取数据

    Params:
    ------
        file_path: 数据文件路径

    Returns:
    -------
        data: DataFrame类型，数据包含'English'和'Chinese'两个表头
    '''
    data = pd.read_table(file_path)
    data.columns = ['English', 'Chinese']
    return data

def build_subwords_tokenizers(train_path, en_filename_prefix, cn_filename_prefix):
    '''
    从训练数据集构建自定义分词器，并将构建好的分词器存储到指定的文件中

    Params:
    ------
        train_path: 训练数据集所在路径
        en_filename_prefix: 英文分词器存储文件名
        cn_filename_prefix: 中文分词器存储文件名
    '''
    train_examples = get_data(train_path)

    # 从训练数据集创建自定义子词分词器
    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus((en for en in train_examples['English']), target_vocab_size=2**13)
    tokenizer_cn = tfds.features.text.SubwordTextEncoder.build_from_corpus((cn for cn in train_examples['Chinese']), target_vocab_size=2**13)
    
    # 将构建好的分词器存储到指定路径中
    tokenizer_en.save_to_file(en_filename_prefix)
    tokenizer_cn.save_to_file(cn_filename_prefix)

def get_tokenizers(en_file_prefix, cn_file_prefix):
    '''
    从指定文件中获取分词器

    Params:
    ------
        en_file_prefix: 英文分词器存储文件名
        cn_file_prefix: 中文分词器存储文件名
    
    Returns:
    -------
        tokenizer_en: 英文分词器
        tokenizer_cn: 中文分词器
    '''
    tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(en_file_prefix)
    tokenizer_cn = tfds.features.text.SubwordTextEncoder.load_from_file(cn_file_prefix)
    return tokenizer_en, tokenizer_cn

def sentence2vec(en_data, cn_data, tokenizer_en, tokenizer_cn):
    '''
    利用tokenizer将中文句子和英文句子转换成向量形式

    Params:
    ------
        cn_data: 中文数据
        en_data: 英文数据
        tokenizer_en: 英文分词器
        tokenizer_cn: 中文分词器

    Returns:
    -------
        cn_vec: 中文数据对应的向量列表
        en_vec: 英文数据对应的向量列表
    '''
    cn_vec = [tokenizer_cn.encode(data) for data in cn_data]
    en_vec = [tokenizer_en.encode(data) for data in en_data]
    return en_vec, cn_vec

def add_token(en_vec, cn_vec, tokenizer_en, tokenizer_cn):
    '''
    在将句子转换成向量之后，还要在向量的开头和结尾加入代表start和end的token，方便模型知道什么时候开始和结束

    Params:
    ------
        en_vec: 英文向量列表
        cn_vec: 中文向量列表
        tokenizer_en: 英文分词器
        tokenizer_cn: 中文分词器

    Returns:
    -------
        full_en_vec: 添加完token的英文向量列表
        full_cn_vec: 添加完token的中文向量列表
    '''
    full_en_vec = []
    full_cn_vec = []
    for vec in en_vec:
        full_en_vec.append([tokenizer_en.vocab_size] + vec + [tokenizer_en.vocab_size+1])
    for vec in cn_vec:
        full_cn_vec.append([tokenizer_cn.vocab_size] + vec + [tokenizer_cn.vocab_size+1])
    return full_en_vec, full_cn_vec

def pad_vectors(en, cn, max_length):
    '''
    在添加完token之后还要将向量补齐成统一的长度，长度不够的在后面补0，过长的就将后面的信息切除

    Params:
    ------
        en: 英文向量列表
        cn: 中文向量列表

    Returns:
    -------
        en_padded: 补齐后的英文向量列表
        cn_padded: 补齐后的中文向量列表
    '''
    en_padded = tfkeras.preprocessing.sequence.pad_sequences(en, maxlen=max_length, dtype='int64', padding='post', value=0.0)
    cn_padded = tfkeras.preprocessing.sequence.pad_sequences(cn, maxlen=max_length, dtype='int64', padding='post', value=0.0)
    return en_padded, cn_padded

def get_batch(en, cn, batch_size):
    '''
    在将向量都补齐到相同长度后，此时数据格式还都是ndarray，因此还需要将其转换为TensorFlow的数据格式
    同时将完整的数据划分为多个batch

    Params:
    ------
        en: 英文数据
        cn: 中文数据
        batch_size: 每个batch的大小

    Returns:
    -------
        en_batch:
        cn_batch:
    '''
    en_batch = tf.data.Dataset.from_tensor_slices(en)
    cn_batch = tf.data.Dataset.from_tensor_slices(cn)
    
    en_batch = en_batch.padded_batch(batch_size)
    cn_batch = cn_batch.padded_batch(batch_size)
    return en_batch, cn_batch