import tensorflow as tf
import tensorflow.keras as tfkeras
import numpy as np

# -------------- Positional Encoding -------------- #
def get_angles(pos, i, d_model):
    '''
    词嵌入是被嵌入到d_model维的空间中，位置编码就是要生成d_model维的位置向量添加到嵌入向量中，
    该函数生成对应每一个维度索引的角度值。

    Params:
    ------
        pos: 词的位置
        i: 维度索引
        d_model: 模型维度

    Returns:
    -------
        pos * angle_rates: 角度值
    '''
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    '''
    由于事先已经计算好了d_model中每个维度对应的角度值，现在只需要根据索引值的奇偶来决定是使用，
    sin还是cos进行编码

    Params:
    ------
        position: 词的位置
        d_model: 模型维度

    Returns:
    -------
        pos_encoding: 位置编码，dtype=tf.float32
    '''
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    # 此时计算得到的angle_rads的维度为(position, d_model)

    # 将sin应用于数组中的偶数索引处：2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 将cos应用于数组中的奇数索引出：2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# -------------- Masking -------------- #
def create_padding_mask(seq):
    '''
    遮挡一批输入序列中所有的填充标记（pad tokens），从而确保模型不会将填充作为输入。
    生成的mask表明填充值0出现的位置，如果是填充值0的话，mask为1；反之，mask为0。

    Params:
    ------
        seq: 输入的一批序列，尺寸为(batch_size, seq_len)
    
    Returns:
    -------
        mask: 输出的mask的尺寸为(batch_size, 1, 1, seq_len)
    '''
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 添加额外的维度来将填充加到注意力对数(logits)
    mask = seq[:, np.newaxis, np.newaxis, :]
    return mask

def create_look_ahead_mask(seq_len):
    '''
    前瞻遮挡用于遮挡一个序列中的后续标记，也就是该mask表明了不应该使用的条目。
    即预测第三个词时，只能使用第一个和第二个词，而不能使用第三个词后面的条目。

    Params:
    ------
        seq_len: 序列长度
    
    Returns:
    -------
        mask: 输出的mask的尺寸为(seq_len, seq_len)
    '''
    mask = 1 - tf.linalg.band_part(tf.ones(shape=(seq_len, seq_len)), -1, 0)
    
    return mask

# -------------- Scaled dot product attention -------------- #
def scaled_dot_product_attention(q, k, v, mask=None):
    '''
    计算点积注意力权重。

    Params:
    ------
        q: Query， 形状 == (..., seq_len_q, depth) 
        k: Key， 形状 == (..., seq_len_k, depth)
        v: Value， 形状 == (..., seq_len_v, depth_v)
        mask: Float张量，其形状可以转换为(..., seq_len_q, seq_len_k)，默认值为None

    Returns:
    -------
        output: 最终的输出张量，有Query与Key计算得来
        attention_weights: 注意力权重张量
    '''
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # 为了获取更加平滑的梯度，对matmul_qk进行缩放
    dk = tf.shape(k)[-1]
    dk = tf.cast(dk, dtype=tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 将mask加入到缩放的张量上
    if mask is not None:
        scaled_attention_logits += (mask * (-1e9))
    
    # 进行softmax操作
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # (..., seq_len_q, seq_len_k)

    # 计算最终输出结果
    output = tf.matmul(attention_weights, v)

    return output, attention_weights

# -------------- Multi-head attention -------------- #
class MultiHeadAttention(tfkeras.layers.Layer):
    def __init__(self, d_model, num_heads):
        pass

    
if __name__ == '__main__':
    pass