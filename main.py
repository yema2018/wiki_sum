import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
from my_model import *
import tensorflow as tf
import time
from data_process import generate_batch
import sentencepiece as spm
from nn_att import PreAtt
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams
import collections

tf.enable_eager_execution()

def parse_args():
    parser = argparse.ArgumentParser(description='Run graph2vec based MDS tasks.')
    parser.add_argument('--mode', nargs='?', default='train', help='must be the val_no_sp/decode')
    parser.add_argument('--ckpt_path', nargs='?', default='./checkpoints/train_large', help='checkpoint path')

    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epoch', type=int, default=4, help='epoch')

    parser.add_argument('--num_layers', type=int, default=5, help='the number of layers in transformer')
    parser.add_argument('--d_model', type=int, default=256, help='the dimension of embedding')
    parser.add_argument('--num_headers', type=int, default=4, help='the number of attention headers')
    parser.add_argument('--dff', type=int, default=1024, help='the number of units in point_wise_feed_forward_network')
    parser.add_argument('--drop_rate', type=float, default=0.3, help='drop rate')

    parser.add_argument('--beam_size', type=int, default=5, help='beam search size.')
    parser.add_argument('--block_n_grams', type=int, default=3, help='prevent generating n grams')
    parser.add_argument('--block_n_words_before', type=int, default=3, help='block n words before the current step')


    return parser.parse_args()


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=50000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class RUN:
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load('spm9998_3.model')

        self.seq2seq = MyModel(args.num_layers, args.d_model, args.num_headers, args.dff, 32000,
                                 25, args.drop_rate)

        learning_rate = CustomSchedule(args.d_model)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9, clipvalue=0.5, clipnorm=1.0)
        # self.optimizer = tf.keras.optimizers.SGD(0.01)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction='none')

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')

        checkpoint_path = args.ckpt_path

        self.ckpt = tf.train.Checkpoint(model=self.seq2seq,
                                      optimizer=self.optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=50)

        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            path = 'checkpoints/train_large/ckpt-30'
            self.ckpt.restore(path)
            print('{} restored!!'.format(path))

        self.attpre = PreAtt()

    def masked_loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def train_step(self, inp, ranks, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            pre, _, _ = self.seq2seq(inp, True, ranks, tar_inp)
            # print(tf.argmax(pre, axis=-1))

            loss = self.masked_loss_function(tar_real, pre)

        gradients = tape.gradient(loss, self.seq2seq.trainable_variables)
        # print(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.seq2seq.trainable_variables))

        self.train_loss(loss)
        self.train_acc(tar_real, pre)
        return tar_real, tf.argmax(pre, axis=-1)

    def saver(self, epoch, start):
        ckpt_save_path = self.ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            self.train_loss.result(),
                                                            self.train_acc.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    def train(self):
        for epoch in range(args.epoch):
            start = time.time()

            self.train_loss.reset_states()
            self.train_acc.reset_states()

            batch_set = generate_batch(args.batch_size, mode=args.mode)
            for (batch, batch_contents) in enumerate(batch_set):
                inp, tar, ranks= batch_contents
                # print(inp.shape, inp_x.shape, ranks.shape, sen_pos.shape,
                # tar.shape, tar_x.shape)
                real, pre = self.train_step(inp, ranks, tar)

                if batch % 50 == 0:
                    print(real)
                    print(pre)
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, self.train_loss.result(), self.train_acc.result()))

                if batch % 20000 == 0 and batch > 0:
                    self.saver(epoch, start)

            self.saver(epoch, start)

    def generate_pw(self):
        batch_set = generate_batch(args.batch_size, mode='train')
        para_weights = np.zeros([1, 25])
        para_embs = np.zeros([1,25,256])
        for (batch, batch_contents) in enumerate(batch_set):
            inp, tar, ranks = batch_contents
            tar_inp = tar[:, :-1]
            _, pw, pb = self.seq2seq(inp, False, ranks, tar_inp, True)
            para_weights = np.concatenate((para_weights, pw.numpy()))
            para_embs = np.concatenate((para_embs, pb.numpy()))
            print(para_embs.shape)

            if len(para_weights) >= 50000:
                break
        np.savetxt('pre_att/pw15', para_weights)
        np.savetxt('pre_att/pb15', para_embs.reshape([-1, 256]))

    def predict_para_att_ratio(self, pb):

        pw = self.attpre.ex_pred(pb)

        return pw.numpy().reshape([-1])

    def score_diversity(self, tar, pred_att_ratio, real_att, beta=0):
        length = int(tar.shape[-1])
        real_att = real_att.numpy()[0, :] / length

        score = 0
        for i in range(25):
            score += np.log(np.minimum(real_att[i], pred_att_ratio[i]))

        # con_sim = cosine_similarity(real_att.reshape(1, -1), pred_att_ratio.reshape(1, -1))[0]

        return beta * score

    def eval_by_beam_search(self):
        batch_set = generate_batch(1, mode='test', para_len=120)
        bng = args.block_n_grams
        bnw = args.block_n_words_before
        for (batch, batch_contents) in enumerate(batch_set):
            inp, tar, ranks = batch_contents
            assert inp.shape[0] == 1

            initial_dec_inp = [4]
            initial_dec_inp = tf.expand_dims(initial_dec_inp, 0)
            output = [[initial_dec_inp, 0]]
            final_out = []
            update_beam = 0

            for step in range(2000):
                temp = []

                bsize = args.beam_size - update_beam

                for out in output:

                    tar_inp = out[0]
                    indices = []
                    if int(tar_inp.shape[-1]) >= bng:
                        tar_inp2 = list(tar_inp.numpy()[0,:])
                        ngram = set([i for i in ngrams(tar_inp2, bng)])
                        for gram in ngram:
                            if tar_inp2[-(bng-1):] == list(gram[:(bng-1)]):
                                indices.append(gram[-1])

                    pre, pw, pb = self.seq2seq(inp, False, ranks, tar_inp, True)

                    pre = pre[:, -1, :]  # (1, vocab_extend)

                    if indices:
                        ind = tf.reshape(tf.constant(indices), [-1, 1])
                        mod = tf.scatter_nd(indices=ind, updates=tf.ones(len(indices)),
                                            shape=tf.constant([pre.shape[-1]]))
                        pre -= mod

                    indices1 = tf.constant(tf.reshape(out[0][0, -bnw:], [-1, 1]))
                    mod1 = tf.scatter_nd(indices=indices1, updates=tf.ones(indices1.shape[0]),
                                         shape=tf.constant([pre.shape[-1]]))
                    pre -= mod1

                    val, index = tf.nn.top_k(pre, bsize)

                    for i in range(bsize):
                        q = tf.concat((out[0], tf.expand_dims([index[0, i]], 0)), axis=-1)

                        p = out[1] + np.log(val[0, i])

                        temp.append([q, p, pw, pb])

                output = sorted(temp, key=lambda x: x[1])[-bsize:]
                print(output[-1][0])

                for o in output.copy():

                    if o[0][0, -1].numpy() == 5:
                        # print(o[0])
                        output.remove(o)
                        pred_pw = self.predict_para_att_ratio(o[-1])
                        diver_score = self.score_diversity(o[0], pred_pw, o[-2])
                        # lp = ((5 + int(o[0].shape[-1])) ** 0.7) / ((5 + 1) ** 0.7)
                        o[1] /= int(o[0].shape[-1])
                        print(o[1], diver_score)
                        o[1] += diver_score

                        final_out.append(o)
                        update_beam += 1

                        # abs1 = list(o[0].numpy().reshape([-1]).astype(int))
                        # abs1 = [int(i) for i in abs1]
                        # out_sen = self.sp.decode_ids(abs1)
                        # with open('./temp3/compare', 'a') as fw:
                        #     fw.write(out_sen)
                        #     fw.write('\t')
                        #     fw.write('prob {:.4f}'.format(o[1]-diver_score))
                        #     fw.write('\t')
                        #     fw.write('diver {:.4f}'.format(diver_score))
                        #     fw.write('\n')

                if not output:
                    break

            if not final_out:
                final_out = output

            final_out = sorted(final_out, key=lambda x: x[1])[-1][0].numpy().reshape([-1]).astype(int)
            abs1 = list(final_out)
            abs1 = [int(i) for i in abs1]
            out_sen = self.sp.decode_ids(abs1)

            with open('./temp3/120_b0_post_30', 'a') as fw:
                fw.write(out_sen)
                fw.write('\n')

    def valid_step(self, inp, ranks, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        pre, _, _ = self.seq2seq(inp, False, ranks, tar_inp)

        loss = self.masked_loss_function(tar_real, pre)
        self.train_loss(loss)

    def valid(self):
        path_range = range(6, 20)
        for path in path_range:
            self.ckpt.restore('checkpoints/train_large/ckpt-{}'.format(path))
            # print()
            start = time.time()
            self.train_loss.reset_states()

            batch_set = generate_batch(128, mode='valid')
            for (batch, batch_contents) in enumerate(batch_set):
                inp, tar, ranks = batch_contents

                _, _ = self.train_step(inp, ranks, tar)

                if batch % 50 == 0:
                    print('50 cal')

            print('ckpt-{} Loss {:.4f}'.format(path, self.train_loss.result()))

            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


if __name__ == "__main__":
    args = parse_args()
    a = RUN()
    a.train()

