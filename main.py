import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
from my_model import *
import tensorflow as tf
import time
from data_process import generate_batch
import sentencepiece as spm
from nltk.util import ngrams

# tf.enable_eager_execution()
#
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1000)]
# )

def parse_args():
    parser = argparse.ArgumentParser(description='Run graph2vec based MDS tasks.')
    parser.add_argument('--mode', nargs='?', default='train', help='must be the val_no_sp/decode')
    parser.add_argument('--model_mode', nargs='?', default='p', help='v: vertical or p: parallel')
    parser.add_argument('--ckpt_path', nargs='?', default='./checkpoints/train_large_p_3d_30', help='checkpoint path')

    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--epoch', type=int, default=4, help='epoch')

    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers in transformer')
    parser.add_argument('--d_model', type=int, default=256, help='the dimension of embedding')
    parser.add_argument('--num_headers', type=int, default=4, help='the number of attention headers')
    parser.add_argument('--dff', type=int, default=1024, help='the number of units in point_wise_feed_forward_network')
    parser.add_argument('--drop_rate', type=float, default=0.3, help='drop rate')

    parser.add_argument('--beam_size', type=int, default=5, help='beam search size.')
    parser.add_argument('--block_n_grams', type=int, default=3, help='prevent generating n grams')
    parser.add_argument('--block_n_words_before', type=int, default=2, help='block n words before the current step')

    return parser.parse_args()


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=16000):
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
                                args.drop_rate, args.model_mode)

        learning_rate = CustomSchedule(args.d_model)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

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
            path = self.ckpt_manager.latest_checkpoint
            self.ckpt.restore(path)
            print('{} restored!!'.format(path))

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
            pre, pw = self.seq2seq(inp, True, ranks, tar_inp)
            # print(tf.argmax(pre, axis=-1))

            loss = self.masked_loss_function(tar_real, pre)

        gradients = tape.gradient(loss, self.seq2seq.trainable_variables)
        # print(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.seq2seq.trainable_variables))

        self.train_loss(loss)
        self.train_acc(tar_real, pre)
        return tar_real, tf.argmax(pre, axis=-1), pw

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
                real, pre, pw = self.train_step(inp, ranks, tar)

                if batch % 50 == 0:
                    print(real)
                    print(pre)
                    # print(pw[0])
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, self.train_loss.result(), self.train_acc.result()))

                if batch % 20000 == 0 and batch > 0:
                    self.saver(epoch, start)

            self.saver(epoch, start)

    def generate_pw(self):
        batch_set = generate_batch(32, para_num=30, para_len=100, mode='test')
        para_weights = np.zeros([1, 30])

        for (batch, batch_contents) in enumerate(batch_set):
            inp, tar, ranks = batch_contents
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            _, pw, _ = self.seq2seq(inp, False, ranks, tar_inp, tar_real, True)
            para_weights = np.concatenate((para_weights, pw.numpy()))

            # if len(para_weights) >= 100:
            #     break
        np.savetxt('pre_att/pw1', para_weights)

    def score_diversity(self, pred_att_ratio, real_att, ranks, beta=0.8):
        real_att = real_att.numpy()[0, :]
        length = np.sum(real_att)
        real_att /= length

        score = 0
        for i in range(int(ranks.shape[-1])):
            if real_att[i] == 0.0:
                continue
            score += np.log(np.minimum(real_att[i], pred_att_ratio[i]))


        return beta * score

    def eval_by_beam_search(self):
        pre_att = PreAtt()
        def predict_para_att_ratio(pb, ranks):
            pw = pre_att.ex_pred(pb, ranks)
            return pw.numpy().reshape([-1])

        batch_set = generate_batch(1, mode='test', para_num=30, para_len=100)
        bng = args.block_n_grams
        bnw = args.block_n_words_before
        for (batch, batch_contents) in enumerate(batch_set):
            inp, tar, ranks = batch_contents
            assert inp.shape[0] == 1

            title = list(inp[0, 0, :])
            try:
                t_id = title.index(3)
                title = [4] + title[:t_id]
                initial_dec_inp = title
            except:
                initial_dec_inp = [4]
            initial_dec_inp = tf.expand_dims(initial_dec_inp, 0)
            output = [[initial_dec_inp, 0]]
            final_out = []
            update_beam = 0

            _, _, pb = self.seq2seq(inp, False, ranks, initial_dec_inp)
            pred_pw = predict_para_att_ratio(pb, ranks)

            # compression
            count_zero = list(ranks[0,:]).count(0)
            fil_len = 25
            fil, fil_id = tf.nn.top_k(pred_pw, fil_len)
            if fil_len > 30 - count_zero:
                count_remove = count_zero - (30 - fil_len)
                fil_id = fil_id[:-count_remove]
            inp = inp[:, fil_id, :]
            ranks = np.ones(shape=(1, len(fil_id)))

            # generation
            for step in range(200):
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

                    pre, pw, _ = self.seq2seq(inp, False, ranks, tar_inp, cal_pw=True)

                    pre = pre[:, -1, :]  # (1, vocab_extend)

                    if indices:
                        ind = tf.reshape(tf.constant(indices), [-1, 1])
                        mod = tf.scatter_nd(indices=ind, updates=tf.ones(len(indices)),
                                            shape=tf.constant([pre.shape[-1]]))
                        pre -= mod  # remove repeated tri-grams

                    bnw_block = list(out[0][0, -(bnw*2):].numpy())
                    while 10 in bnw_block:
                        bnw_block.remove(10)
                    bnw_block = bnw_block[-bnw:]

                    indices1 = tf.constant(tf.reshape(bnw_block, [-1, 1]))
                    mod1 = tf.scatter_nd(indices=indices1, updates=tf.ones(indices1.shape[0]),
                                         shape=tf.constant([pre.shape[-1]]))
                    pre -= mod1  # remove the same tokes within 2 steps

                    val, index = tf.nn.top_k(pre, bsize)

                    for i in range(bsize):
                        q = tf.concat((out[0], tf.expand_dims([index[0, i]], 0)), axis=-1)

                        p = out[1] + np.log(val[0, i])
                        temp.append([q, p])

                output = sorted(temp, key=lambda x: x[1])[-bsize:]
                print(output[-1][0])

                for o in output.copy():

                    if o[0][0, -1].numpy() == 5:

                        output.remove(o)
                        _, pw, pb = self.seq2seq(inp, False, ranks, o[0][:, :-1], cal_pw=True)
                        pred_pw = predict_para_att_ratio(pb, ranks)
                        diver_score = self.score_diversity(pred_pw, pw, ranks)

                        o[1] /= int(o[0].shape[-1])
                        o[1] += diver_score

                        final_out.append(o)
                        update_beam += 1

                if not output:
                    break

            if not final_out:
                final_out = output

            final_out = sorted(final_out, key=lambda x: x[1])[-1][0].numpy().reshape([-1]).astype(int)
            abs1 = list(final_out)
            abs1 = [int(i) for i in abs1]
            out_sen = self.sp.decode_ids(abs1)

            with open('b_0.8_f25_nd.txt', 'a',encoding='utf8') as fw:
                fw.write(out_sen)
                fw.write('\n')


    def valid_step(self, inp, ranks, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        pre, _, = self.seq2seq(inp, False, ranks, tar_inp)

        loss = self.masked_loss_function(tar_real, pre)
        self.train_loss(loss)

    def valid(self):
        path_range = range(1, 14)
        for path in path_range:
            self.ckpt.restore('checkpoints/train_large_st_1d_s_25/ckpt-{}'.format(path))
            print('ckpt-{} restored'.format(path))
            start = time.time()
            self.train_loss.reset_states()

            batch_set = generate_batch(32, mode='valid')
            for (batch, batch_contents) in enumerate(batch_set):
                inp, tar, ranks = batch_contents

                self.valid_step(inp, ranks, tar)

                # if batch % 50 == 0:
                     # print('50 batches cal')

            print('ckpt-{} Loss {:.4f}'.format(path, self.train_loss.result()))

            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


"""
=======================================================================================================================
"""


class PreAttModel(tf.keras.Model):
    def __init__(self, layers=2, d_model=256, num_heads=4, dff=1024, rate=0.5):
        super(PreAttModel, self).__init__()
        self.layer = layers
        self.m = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(layers)]
        self.out_layer = tf.keras.layers.Dense(1)

    def call(self, inp, training, ranks):
        batch = tf.shape(inp)[0]
        mask = create_padding_mask(ranks)
        h = inp
        for i in range(self.layer):
            h = self.m[i](h, training, mask)

        h = self.out_layer(h)
        logits = tf.reshape(h, shape=[batch, -1])
        mask = tf.reshape(mask, [batch, -1])
        logits += mask * -1e19
        att_ratio = tf.nn.softmax(logits, axis=1)

        return att_ratio    # shape == (batch_size, para_num)


class PreAtt(object):
    def __init__(self):
        self.model = PreAttModel()

        self.optimizer = tf.keras.optimizers.Adam()
        # self.optimizer = tf.keras.optimizers.SGD(0.01)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')

        checkpoint_path = './checkpoints2/3d_l'

        ckpt = tf.train.Checkpoint(model=self.model,
                                   optimizer=self.optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=30)

        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            path = self.ckpt_manager.latest_checkpoint
            ckpt.restore(path)
            print('{} restored!!'.format(path))


    def seq2seq_model(self):

        return RUN().seq2seq

    def cal_loss(self, pre, real):
        pre = tf.reshape(pre, [-1, 1])
        real = tf.reshape(real, [-1, 1])

        loss_ = tf.keras.losses.mse(real, pre)

        return tf.reduce_mean(loss_) * tf.constant([100], tf.float32)

    def train_step(self, inp, real, ranks):
        with tf.GradientTape() as tape:
            pre = self.model(inp, True, ranks)

            loss = self.cal_loss(pre, real)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        # print(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)

    def val_step(self, inp, real, ranks):
        pre = self.model(inp, False, ranks)

        loss = self.cal_loss(pre, real)

        self.val_loss(loss)

    def train(self):
        seq2seq = self.seq2seq_model()
        for epoch in range(30):
            start = time.time()

            self.train_loss.reset_states()
            self.val_loss.reset_states()
            print('start training')
            batch_set = generate_batch(32)
            for (batch, batch_contents) in enumerate(batch_set):
                inp, tar, ranks = batch_contents
                tar_inp = tar[:, :-1]
                tar_real = tar[:, 1:]
                _, pw, pb = seq2seq(inp, False, ranks, tar_inp, tar_real, True)

                pw /= tf.reduce_sum(pw, axis=1, keepdims=True)

                self.train_step(pb, pw, ranks)

                if batch % 50 == 0 :
                    print('Epoch {} Batch {} Loss {:.4f}'.format(
                        epoch + 1, batch, self.train_loss.result()))

            ckpt_save_path = self.ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

            print('Epoch {} Loss {:.4f}'.format(epoch + 1, self.train_loss.result()))
            print('\nstart validation')
            val_batch = generate_batch(32, mode='valid')
            for (b, bc) in enumerate(val_batch):
                inp, tar, ranks = bc
                tar_inp = tar[:, :-1]
                tar_real = tar[:, 1:]
                _, pw, pb = seq2seq(inp, False, ranks, tar_inp, tar_real, True)

                pw /= tf.reduce_sum(pw, axis=1, keepdims=True)
                self.val_step(pb, pw, ranks)

            print('Validation: Loss {:.4f}'.format(self.val_loss.result()))

            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    def valid(self):
        seq2seq = self.seq2seq_model()
        val_batch = generate_batch(32, mode='valid')
        for (b, bc) in enumerate(val_batch):
            inp, tar, ranks = bc
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            _, pw, pb = seq2seq(inp, False, ranks, tar_inp, tar_real, True)

            pw /= tf.reduce_sum(pw, axis=1, keepdims=True)
            self.val_step(pb, pw, ranks)

        print('Validation: Loss {:.4f}'.format(self.val_loss.result()))

    def predict(self):
        seq2seq = self.seq2seq_model()
        val_batch = generate_batch(32, mode='test')
        t = np.zeros([1, 30])
        ty = np.zeros([1, 30])
        for (b, bc) in enumerate(val_batch):
            if b > 5:
                break
            inp, tar, ranks = bc
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            _, pw, pb = seq2seq(inp, False, ranks, tar_inp, tar_real, True)

            pw /= tf.reduce_sum(pw, axis=1, keepdims=True)
            pre = self.model(pb, False, ranks)

            t = np.concatenate((t, pre.numpy() * 100))
            ty = np.concatenate((ty, pw.numpy() * 100))


        np.savetxt('pre_att/pre1', t)
        np.savetxt('pre_att/att1', ty)

    def ex_pred(self, inp, ranks):

        return self.model(inp, False, ranks)


if __name__ == "__main__":
    args = parse_args()
    a = RUN()
    a.generate_pw()

    #b = PreAtt()
    #b.train()