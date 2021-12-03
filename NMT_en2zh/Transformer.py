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
    output = tf.matmul(attention_weights, v) # (..., seq_len_q, depth_v)

    return output, attention_weights

# -------------- Multi-head attention -------------- #
class MultiHeadAttention(tfkeras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # 由于是对头部的维度进行拆分，所以要保证d_model能够整除num_heads
        assert d_model % num_heads == 0
        
        self.depth = self.d_model // self.num_heads

        self.wq = tfkeras.layers.Dense(self.d_model)
        self.wk = tfkeras.layers.Dense(self.d_model)
        self.wv = tfkeras.layers.Dense(self.d_model)

        self.dense = tfkeras.layers.Dense(self.d_model)
    
    def split_heads(self, x, batch_size):
        '''
        注意力头部拆分，拆分的方法就是通过reshape将输入x的形状变为(batch_size, seq_len, num_heads, depth)
        最后输出结果的形状通过转置变为(batch_size, num_heads, seq_len, depth)

        Params:
        ------
            x: 输入张量的形状为(batch_size, seq_len, d_model)
            batch_size: 批量大小
        
        Returns:
        -------
            tf.transpose(x)
        '''
        x = tf.reshape(x, shape=(batch_size, -1, self.num_heads, self.depth))
        # 此时x的形状为(batch_size, seq_len, num_heads, depth)
        # 还要通过转置将x的形状变为(batch_size, num_heads, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)    # (batch_size, seq_len, d_model)
        k = self.wk(k)    # (batch_size, seq_len, d_model)
        v = self.wv(v)    # (batch_size, seq_len, d_model)

        # 开始拆分成多头
        q = self.split_heads(q, batch_size)    # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)    # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)    # (batch_size, num_heads, seq_len_v, depth)

        # 计算多头注意力
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # 将多头注意力合并
        # 将scaled_attention的形状变为(batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # concat_attention.shape == (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, shape=(batch_size, -1, self.d_model))

        # 再过一个全连接层
        # output.shape == (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)
        return output, attention_weights

# -------------- Point wise feed forward network -------------- #
def point_wise_feed_forward_network(d_model, dff):
    '''
    创建一个含有两个子层的点式前馈网络，第一个子层使用ReLU激活函数。

    Params:
    ------
        d_model: 前馈网络最终的输出维度
        dff: 第一个子层的输出维度

    Returns:
    ------
        ffn: 一个具有两层子层的前馈网络
    '''
    ffn = tfkeras.Sequential([
        tfkeras.layers.Dense(dff, activation='relu'),   # (batch_size, seq_len, dff)
        tfkeras.layers.Dense(d_model)    # (batch_size, seq_len, d_model)
    ])
    return ffn

# -------------- Encoder & Decoder -------------- #

    
if __name__ == '__main__':
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))
    out, attn = temp_mha(y, y, y)
    print(out.shape, attn.shape)