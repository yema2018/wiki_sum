import torch
import tensorflow as tf
import os
import numpy as np
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('spm9998_3.model')


def generate_batch(batch_size, para_num=30, para_len=100, tgt_len=145, mode='train'):
    count = 0
    l = [i for i in os.listdir('ranked_wiki_b40') if mode in i]

    for i in l:
        print(i)
        da = torch.load('ranked_wiki_b40/{}'.format(i))
        srcl = []
        tgtl = []
        ranks = []
        for d in da:
            tgt = d['tgt']
            src = d['src'][:para_num]

            src = tf.keras.preprocessing.sequence.pad_sequences(src, maxlen=para_len, padding='post', truncating='post'
                                                                , value=0)
            rank = np.array(range(1, len(src) + 1))

            if len(src) != para_num:
                pad_lines = para_num-len(src)
                src = np.pad(src, ((0, pad_lines), (0, 0)), 'constant', constant_values=0)
                rank = np.pad(rank, (0, pad_lines), 'constant', constant_values=0)

            tgt = tf.keras.preprocessing.sequence.pad_sequences([tgt],
                                                                maxlen=tgt_len, padding='post', truncating='post',
                                                                value=0)

            srcl.append(np.expand_dims(src, 0))
            tgtl.append(tgt)
            ranks.append(np.expand_dims(rank, 0))

        srcl = np.concatenate(srcl, axis=0)
        tgtl = np.concatenate(tgtl, axis=0)
        ranks = np.concatenate(ranks, axis=0)

        batch_num = int(np.ceil(len(ranks) / batch_size))

        for i in range(batch_num):
            # count += 1
            # if count <= 12956:
            #     continue
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(ranks))
            s = srcl[start_index: end_index]  # (batch_size, node_num, inp_seq_len)
            t = tgtl[start_index: end_index]  # (batch_size, node_num)
            r = ranks[start_index: end_index]

            yield s, t, r

