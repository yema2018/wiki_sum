import os

import tensorflow as tf
import numpy as np
from sen_encoder import EncoderLayer, create_padding_mask
from sklearn.model_selection import train_test_split
import time
from data_process import generate_batch
import main



# def load_data(pb='pre_att/pb', pw='pre_att/pw', ranks='pre_att/ranks'):
#     para_embs = np.genfromtxt(pb).reshape([-1, 25, 256])[1:].astype(np.float32)
#     para_w = np.genfromtxt(pw)[1:].astype(np.float32)
#     para_w /= np.sum(para_w, axis=1, keepdims=True)
#     ranks = np.genfromtxt(ranks)[1:]
#
#     da_split = int(ranks.shape[0] * 0.2)
#     train_x = para_embs[:-da_split]
#     test_x = para_embs[-da_split:]
#     train_y = para_w[:-da_split]
#     test_y = para_w[-da_split:]
#     train_rank = ranks[:-da_split]
#     test_rank = ranks[-da_split:]
#
#     return train_x, test_x, train_y, test_y, train_rank, test_rank
#
#
# def generate_batch(trax, tx, tray, ty, trar, tr, batch_size=64, isTrain=True):
#     if isTrain:
#         batch_num = int(np.ceil(len(trax) / batch_size))
#         for i in range(batch_num):
#             start_index = i * batch_size
#             end_index = min((i + 1) * batch_size, len(trax))
#             inp = trax[start_index: end_index]  # (batch_size, node_num, inp_seq_len)
#             tgt = tray[start_index: end_index]  # (batch_size, node_num)
#             rank = trar[start_index: end_index]
#             yield inp, tgt, rank
#
#     else:
#         batch_num = int(np.ceil(len(tx) / batch_size))
#         for i in range(batch_num):
#             start_index = i * batch_size
#             end_index = min((i + 1) * batch_size, len(tx))
#             inp = tx[start_index: end_index]  # (batch_size, node_num, inp_seq_len)
#             tgt = ty[start_index: end_index]  # (batch_size, node_num)
#             rank = tr[start_index: end_index]
#             yield inp, tgt, rank


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

        self.optimizer2 = tf.keras.optimizers.Adam()
        # self.optimizer = tf.keras.optimizers.SGD(0.01)

        self.train_loss2 = tf.keras.metrics.Mean(name='train_loss2')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')

        checkpoint_path = './checkpoints2/3d'

        ckpt = tf.train.Checkpoint(model=self.model,
                                   optimizer=self.optimizer2)

        self.ckpt_manager2 = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=30)

        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager2.latest_checkpoint:
            path = self.ckpt_manager2.latest_checkpoint
            ckpt.restore(path)
            print('{} restored!!'.format(path))

        self.seq2seq = main.RUN().seq2seq

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
        self.optimizer2.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss2(loss)

    def val_step(self, inp, real, ranks):
        pre = self.model(inp, False, ranks)

        loss = self.cal_loss(pre, real)

        self.val_loss(loss)

    def train(self):

        for epoch in range(30):
            start = time.time()

            self.train_loss2.reset_states()
            self.val_loss.reset_states()
            print('start training')
            batch_set = generate_batch(32, nn_att=True)
            for (batch, batch_contents) in enumerate(batch_set):
                inp, tar, ranks = batch_contents
                tar_inp = tar[:, :-1]
                tar_real = tar[:, 1:]
                _, pw, pb = self.seq2seq(inp, False, ranks, tar_inp, tar_real, True)

                self.train_step(pb, pw, ranks)

                # if batch % 1000 == 0 and batch > 0:
                #     print('Epoch {} Batch {} Loss {:.4f}'.format(
                #         epoch + 1, batch, self.train_loss.result()))
                #     print('\nstart validation\n')
                #     val_batch = generate_batch(trx, tx, tray, ty, isTrain=False)
                #     for(b, bc) in enumerate(val_batch):
                #         self.val_step(bc[0], bc[1])
                #     print('Validation: Loss {:.4f}'.format(self.val_loss.result()))

            ckpt_save_path = self.ckpt_manager2.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

            print('Epoch {} Loss {:.4f}'.format(epoch + 1, self.train_loss2.result()))
            print('\nstart validation')
            val_batch = generate_batch(32, nn_att=True, att_train=False)
            for (b, bc) in enumerate(val_batch):
                inp, tar, ranks = bc
                tar_inp = tar[:, :-1]
                tar_real = tar[:, 1:]
                _, pw, pb = self.seq2seq(inp, False, ranks, tar_inp, tar_real, True)

                self.val_step(pb, pw, ranks)

            print('Validation: Loss {:.4f}'.format(self.val_loss.result()))

            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    def predict(self):
        val_batch = generate_batch(32, nn_att=True, att_train=False)
        t = np.zeros([1, 25])
        ty = np.zeros([1, 25])
        for (b, bc) in enumerate(val_batch):
            inp, tar, ranks = bc
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            _, pw, pb = self.seq2seq(inp, False, ranks, tar_inp, tar_real, True)

            pre = self.model(pb, False, ranks)

            t = np.concatenate((t, pre.numpy() * 100))
            ty = np.concatenate((ty, pw.numpy() * 100))

        np.savetxt('pre_att/pre1', t)
        np.savetxt('pre_att/att1', ty * 100)

    def ex_pred(self, inp, ranks):

        return self.model(inp, False, ranks)


def eval_ranks(real, pre, num):
    assert num <= real.shape[-1]
    para_num = real.shape[-1]
    aver = 0
    for r, p in zip(real, pre):
        dicr = dict()
        dicp = dict()
        count = 0
        for i in range(para_num):
            dicr[i] = r[i]
            dicp[i] = p[i]

        re = sorted(dicr.items(), key=lambda x: x[-1], reverse=True)
        re = [i[0] for i in re][:num]
        pr = sorted(dicp.items(), key=lambda x: x[-1], reverse=True)
        pr = [i[0] for i in pr][:num]

        for i in range(num):
            if re[i] == pr[i]:
                count += 1

        recall = count/num
        aver += recall

    return aver/len(pre)

def eval_mae(real, pre, thresold):
    para_num = real.shape[-1]
    aver = 0
    for r, p in zip(real, pre):
        count = 0
        for i in range(para_num):
            mae = np.abs(r[i] - p[i]) / r[i]
            if mae < thresold:
                count += 1
        aver += count/para_num

    return aver/len(pre)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    PreAtt().train()
    #
    # pre = np.genfromtxt('pre_att/pre1')[1:]
    # real = np.genfromtxt('pre_att/att1')
    # print(eval_mae(real, pre, 0.2))


