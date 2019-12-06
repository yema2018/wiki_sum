import os

import tensorflow as tf
import numpy as np
from sen_encoder import EncoderLayer
from sklearn.model_selection import train_test_split
import time
tf.enable_eager_execution()


def load_data(pb='pre_att/pb15', pw='pre_att/pw15'):
    para_embs = np.genfromtxt(pb).reshape([-1, 25, 256])[1:].astype(np.float32)
    para_w = np.genfromtxt(pw)[1:].astype(np.float32)
    para_w /= np.sum(para_w, axis=1, keepdims=True)

    return train_test_split(para_embs, para_w, test_size=0.2, random_state=5)


def generate_batch(trx, tx, tray, ty, batch_size=64, isTrain=True):
    if isTrain:
        batch_num = int(np.ceil(len(trx) / batch_size))
        for i in range(batch_num):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(trx))
            inp = trx[start_index: end_index]  # (batch_size, node_num, inp_seq_len)
            tgt = tray[start_index: end_index]  # (batch_size, node_num)
            yield inp, tgt

    else:
        batch_num = int(np.ceil(len(tx) / batch_size))
        for i in range(batch_num):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(tx))
            inp = tx[start_index: end_index]  # (batch_size, node_num, inp_seq_len)
            tgt = ty[start_index: end_index]  # (batch_size, node_num)
            yield inp, tgt


class PreAttModel(tf.keras.Model):
    def __init__(self, layers=2, d_model=256, num_heads=4, dff=1024, rate=0.5):
        super(PreAttModel, self).__init__()
        self.layer = layers
        self.m = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(layers)]
        self.out_layer2 = tf.keras.layers.Dense(25, activation=tf.nn.softmax)
        self.drop2 = tf.keras.layers.Dropout(rate)

    def call(self, inp, training, mask=None):
        batch = tf.shape(inp)[0]
        h = inp
        for i in range(self.layer):
            h = self.m[i](h, training, mask)

        h = tf.reshape(h, [batch, -1])
        # h = self.out_layer(h)
        # logits = tf.reduce_sum(h, axis=-1)
        h = self.drop2(h, training=training)
        att_ratio = self.out_layer2(h)

        return att_ratio    # shape == (batch_size, para_num)


class PreAtt(object):
    def __init__(self):
        self.model = PreAttModel()

        self.optimizer = tf.keras.optimizers.Adam()
        # self.optimizer = tf.keras.optimizers.SGD(0.01)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')

        checkpoint_path = './checkpoints2/t2_f'

        ckpt = tf.train.Checkpoint(model=self.model,
                                   optimizer=self.optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=20)

        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            path = self.ckpt_manager.latest_checkpoint
            ckpt.restore(path)
            print('{} restored!!'.format(path))

    def cal_loss(self, pre, real):
        pre = tf.reshape(pre, [-1, 1])
        real = tf.reshape(real, [-1, 1])

        loss_ = tf.keras.losses.mse(real, pre)

        return tf.reduce_mean(loss_) * tf.constant([100], tf.float32)

    def train_step(self, inp, real):
        with tf.GradientTape() as tape:
            pre = self.model(inp, True)

            loss = self.cal_loss(pre, real)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        # print(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)

    def val_step(self, inp, real):
        pre = self.model(inp, False)

        loss = self.cal_loss(pre, real)

        self.val_loss(loss)

    def train(self):
        trx, tx, tray, ty = load_data()
        for epoch in range(30):
            start = time.time()

            self.train_loss.reset_states()
            self.val_loss.reset_states()
            print('start training')
            batch_set = generate_batch(trx, tx, tray, ty)
            for (batch, batch_contents) in enumerate(batch_set):
                inp, tar = batch_contents

                self.train_step(inp, tar)

                # if batch % 1000 == 0 and batch > 0:
                #     print('Epoch {} Batch {} Loss {:.4f}'.format(
                #         epoch + 1, batch, self.train_loss.result()))
                #     print('\nstart validation\n')
                #     val_batch = generate_batch(trx, tx, tray, ty, isTrain=False)
                #     for(b, bc) in enumerate(val_batch):
                #         self.val_step(bc[0], bc[1])
                #     print('Validation: Loss {:.4f}'.format(self.val_loss.result()))

            ckpt_save_path = self.ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

            print('Epoch {} Loss {:.4f}'.format(epoch + 1, self.train_loss.result()))
            print('\nstart validation')
            val_batch = generate_batch(trx, tx, tray, ty, isTrain=False)
            for (b, bc) in enumerate(val_batch):
                self.val_step(bc[0], bc[1])

            print('Validation: Loss {:.4f}'.format(self.val_loss.result()))

            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    def predict(self):
        trx, tx, tray, ty = load_data()
        val_batch = generate_batch(trx, tx, tray, ty, isTrain=False)
        t = np.zeros([1, 25])
        for (b, bc) in enumerate(val_batch):
            pre = self.model(bc[0], False)

            t = np.concatenate((t, pre.numpy() * 100))
            print(t.shape)

        np.savetxt('pre_att/pre1', t)
        np.savetxt('pre_att/att1', ty * 100)

    def ex_pred(self, inp):

        return self.model(inp, False)



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    PreAtt().predict()

