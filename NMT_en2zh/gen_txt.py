"""
生成两个txt文件，一份文件存储原始译文（references），另一份存储Transformer翻译出来的译文（candidates）。
"""
from testModel import translate
from preprocess import *
import pandas as pd
import os
from tqdm import tqdm

test_data = get_data(TEST_PATH)

def gen_references_txt(output_dir, file_name):
    cn_data = test_data['Chinese']
    with open(os.path.join(output_dir, file_name), mode='w', encoding='utf-8') as f:
        for sentence in cn_data:
            f.write(sentence + '\n')
    print('References file generated successfully!')

def gen_candidates_txt(output_dir, file_name):
    en_data = test_data['English']
    en_data = tqdm(en_data)
    with open(os.path.join(output_dir, file_name), mode='w', encoding='utf-8') as f:
        for sentence in en_data:
            predict_sentence = translate(sentence)
            f.write(predict_sentence + '\n')
    print('Candidates file generated successfully!')

if __name__ == '__main__':
    # gen_references_txt('./NMT_en2zh/data', 'references.txt')
    gen_candidates_txt('./NMT_en2zh/data', 'candidates.txt')