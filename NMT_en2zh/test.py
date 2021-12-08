import tensorflow as tf
import tensorflow.keras as tfkeras
from train import create_masks
from Transformer import *
from preprocess import *

tokenizer_en, tokenizer_cn = get_tokenizers(EN_FILENAME_PREFIX, CN_FILENAME_PREFIX)
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
input_vocab_size = tokenizer_en.vocab_size + 2
target_vocab_size = tokenizer_cn.vocab_size + 2
dropout_rate = 0.1
MAX_LENGTH = 40

# --------------- Load Model --------------- #
transformer = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, 
                          input_vocab_size=input_vocab_size, target_vocab_size=target_vocab_size, 
                          pe_input=input_vocab_size, pe_target=target_vocab_size, dropout_rate=dropout_rate)
optimizer = tfkeras.optimizers.Adam()

checkpoint_path = '/Users/victor/Desktop/ML_Lab/NMT_en2zh/Checkpoints/train'
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))

def evaluate(inp_sentence):
    # 由于输入是英文语句，所以开始与结束标记如下所示：
    start_token = [tokenizer_en.vocab_size]
    end_token = [tokenizer_en.vocab_size + 1]

    # 给输入语句增加开始与结束标记
    inp_sentence = start_token + tokenizer_en.encode(inp_sentence) + end_token
    # 给句子向量增加一个batch_size维度
    encoder_input = tf.expand_dims(inp_sentence, 0)
    # 将句子向量做padding来补齐
    encoder_input = tfkeras.preprocessing.sequence.pad_sequences(encoder_input, maxlen=MAX_LENGTH, dtype='int64', padding='post', value=0)
    # 将padding后的句子向量转换成Tensor类型
    encoder_input = tf.cast(encoder_input, dtype=tf.int64)

    # 由于目标是中文，所以输入transformer的第一个词应该是中文的开始标记
    decoder_input = [tokenizer_cn.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        predictions, attention_weights = transformer(encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)
        predictions = predictions[:, -1:, :]
        predict_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predict_id == tokenizer_cn.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights
        
        output = tf.concat([output, predict_id], axis=-1)
    return tf.squeeze(output, axis=0), attention_weights

def translate(sentence):
    result, attention_weights = evaluate(sentence)

    predict_sentence = tokenizer_cn.decode([i for i in result if i < tokenizer_cn.vocab_size])
    print(f'Input: {sentence}')
    print(f'Predicted translation: {predict_sentence}')

if __name__ == '__main__':
    translate("What's the weather like today?")