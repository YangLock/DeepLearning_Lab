import tensorflow as tf
import tensorflow.keras as tfkeras
from Transformer import *
from preprocess import *

# ---------------- Get Data ---------------- #
tokenizer_en, tokenizer_cn = get_tokenizers(EN_FILENAME_PREFIX, CN_FILENAME_PREFIX)
train_data = get_data(TRAIN_PATH)

# ---------------- Hyperparameters ---------------- #
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
input_vocab_size = tokenizer_en.vocab_size + 2
target_vocab_size = tokenizer_cn.vocab_size + 2
dropout_rate = 0.1

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
train_accuracy = tfkeras.metrics.SparseCategoricalCrossentropy(name='train_accuracy')

if __name__ == '__main__':
    real = [9, 12, 0, 31, 0, 0, .31, 0, 0, 0, 19, 15, 1, 1]
    flags = tf.math.equal(real, 0)
    mask = tf.math.logical_not(flags)
    mask = tf.cast(mask, dtype=tf.float32)
    print(flags)
    print(mask)