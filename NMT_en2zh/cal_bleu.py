import jieba
from nltk.translate.bleu_score import sentence_bleu
import argparse

def arg_parser():
    Argparser = argparse.ArgumentParser()
    Argparser.add_argument('--reference', type=str, default='./NMT_en2zh/data/references.txt', help='Path of reference file')
    Argparser.add_argument('--candidate', type=str, default='./NMT_en2zh/data/candidates.txt', help='Path of candidate file')
    args = Argparser.parse_args()
    return args

args = arg_parser()

ref_file = args.reference
cad_file = args.candidate

references = []
candidates = []

with open(ref_file, mode='r', encoding='utf-8') as f:
    sentences = f.readlines()
    for sentence in sentences:
        sentence = sentence.strip()
        references.append(sentence)
with open(cad_file, mode='r', encoding='utf-8') as f:
    sentences = f.readlines()
    for sentence in sentences:
        sentence = sentence.strip()
        candidates.append(sentence)

if len(references) != len(candidates):
    raise ValueError('The number of sentences in both files do not match!')

score = 0.0

for ref, cad in zip(references, candidates):
    ref_list = jieba.lcut(ref)
    cad_list = jieba.lcut(cad)
    score += sentence_bleu(references=[ref_list], hypothesis=cad_list)

score /= len(references)
print('-----------------------------')
print(f"The bleu score is: {score}")
print('-----------------------------')