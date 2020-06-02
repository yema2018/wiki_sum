from rouge import Rouge
import torch
import sentencepiece as spm
import os
import matplotlib.pyplot as plt

sp = spm.SentencePieceProcessor()
sp.load('spm9998_3.model')


def eval(sum_path):
    hyp = [i.strip('\n') for i in open(sum_path)]
    l = [i for i in os.listdir('ranked_wiki_b40') if 'test' in i]
    ref = []
    for i in l:
        ref.extend(torch.load('ranked_wiki_b40/{}'.format(i)))

    ref1 = [sp.decode_ids(i['tgt']) for i in ref]
    ref1 = ref1[:len(hyp)]

    rouge = Rouge()
    return rouge.get_scores(hyp, ref1, True)


if __name__ == "__main__":
    print(eval('summary.txt'))