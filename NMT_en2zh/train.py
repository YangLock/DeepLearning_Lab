import time
import tensorflow as tf
import tensorflow.keras as tfkeras
from Transformer import *
from preprocess import *

# ---------------- Get Data ---------------- #
tokenizer_en, tokenizer_cn = get_tokenizers(EN_FILENAME_PREFIX, CN_FILENAME_PREFIX)

# ---------------- Hyperparameters ---------------- #
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
input_vocab_size = tokenizer_en.vocab_size + 2
target_vocab_size = tokenizer_cn.vocab_size + 2
dropout_rate = 0.1
MAX_LENGTH = 40
BATCH_SIZE = 64
EPOCHS = 20    # 训练迭代次数

# ---------------- Optimizer ---------------- #
class CustomSchedule(tfkeras.optimizers.schedules.LearningRateSchedule):
    '''
    按照论文中的公式自定义学习率调度
    '''
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** (-1.5))

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)
optimizer = tfkeras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-8)

# ---------------- Loss & Metrics ---------------- #
loss_object = tfkeras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))    # 由于target序列也都是padding过的，所以要将填充为0的那部分筛选出来，填充值部分是不参与损失值计算的
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

train_loss = tfkeras.metrics.Mean(name='train_loss')
train_accuracy = tfkeras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# ---------------- Training & Checkpointing ---------------- #

# 先初始化一个Transformer模型
transformer = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                          input_vocab_size=input_vocab_size, target_vocab_size=target_vocab_size,
                          pe_input=input_vocab_size, pe_target=target_vocab_size, dropout_rate=dropout_rate)

# 创建checkpoints
checkpoint_path = './Checkpoints/train'
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# 如果checkpoints不是空，就恢复成最新的checkpoint
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!')

def create_masks(inp, tar):
    '''
    Params:
    ------
        inp: 输入序列
        tar: 目标序列
    
    Returns:
    -------
        enc_padding_mask: 编码器所需要的填充遮挡
        combined_mask: 解码器第一个注意力模块所需要的遮挡（填充与前瞻）
        dec_padding_mask: 解码器第二个注意力模块所需要的遮挡（填充）
    '''
    # 生成编码器需要的填充遮挡
    enc_padding_mask = create_padding_mask(inp)

    # 生成解码器的第二个注意力模块所需要的填充遮挡
    dec_padding_mask = create_padding_mask(inp)

    # 生成解码器的第一个注意力模块所需要的填充和前瞻遮挡
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

def get_dataset():
    train_data = get_data(TRAIN_PATH)
    en_data = train_data['English']
    cn_data = train_data['Chinese']
    en_vec, cn_vec = sentence2vec(en_data, cn_data, tokenizer_en, tokenizer_cn)
    full_en_vec, full_cn_vec = add_token(en_vec, cn_vec, tokenizer_en, tokenizer_cn)
    en_padded, cn_padded = pad_vectors(full_en_vec, full_cn_vec, max_length=MAX_LENGTH)
    en_batch, cn_batch = get_batch(en_padded, cn_padded, batch_size=BATCH_SIZE)

    return (en_batch, cn_batch)

train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                        tf.TensorSpec(shape=(None, None), dtype=tf.int64)]
@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp=inp, tar=tar_inp, training=True, enc_padding_mask=enc_padding_mask,
                                     look_ahead_mask=combined_mask, dec_padding_mask=dec_padding_mask)
        loss = loss_function(tar_real, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(tar_real, predictions)

def train(en_batch, cn_batch):
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> English, tar -> Chinese
        for (batch, (inp, tar)) in enumerate(zip(en_batch, cn_batch)):
            train_step(inp, tar)

            if batch % 50 == 0:
                print(f'Epoch {epoch+1} Batch {batch} Loss: {train_loss.result():.4} Accuracy: {train_accuracy.result():.4}')

        if (epoch+1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')
        print(f'Epoch {epoch+1} Loss: {train_loss.result():.4} Accuracy: {train_accuracy.result():.4}')
        print(f'Time taken for 1 epoch: {time.time() - start} secs')

if __name__ == '__main__':
    en_batch, cn_batch = get_dataset()
    train(en_batch, cn_batch)