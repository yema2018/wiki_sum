from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
from sen_encoder import *


def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


def cal_past_att(att_dists):
    shape = att_dists.shape
    past_att = tf.constant(0, tf.float32, [shape[0], 1, shape[-1]])
    for i in range(shape[1] - 1):
        temp = tf.reduce_sum(att_dists[:, :i + 1, :], axis=1, keepdims=True)
        past_att = tf.concat((past_att, temp), axis=1)
    return past_att


class Encoder(tf.keras.layers.Layer):
    def __init__(self, word_enc_layer, d_model, num_heads, dff, input_vocab_size,
                 w_emb, rate):
        super(Encoder, self).__init__()

        self.para_encoder = SenEncoder(word_enc_layer, d_model, num_heads, dff, input_vocab_size, w_emb, rate)

        self.d_model = d_model

    def call(self, inp, training, ranks):
        shape = inp.shape
        para_num = shape[1]

        inp = tf.reshape(inp, shape=[-1, shape[-1]])  # shape == (batch_size * para_num, inp_seq_len)
        padding_mask_l = create_padding_mask(inp)

        padding_mask = create_padding_mask(inp)
        output_mask = create_output_mask(inp)

        # (batch_size * para_num, d_model), (batch_size * para_num, inp_seq_len, d_model)
        con_words = self.para_encoder(inp, training, padding_mask, output_mask)

        return con_words, padding_mask_l, para_num


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2_g = MultiHeadAttention(d_model, num_heads)
        self.mha2_l = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

        self.d_model = d_model

    def call(self, x, para_num, enc_l, training, look_ahead_mask, padding_mask_g, padding_mask_l):
        """
        :param enc_g: shape == (batch_size, para_num, d_model)
        :param enc_l: shape == (batch_size * para_num, inp_seq_len, d_model)

        """
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        shape = tf.shape(attn1)

        out1r = tf.tile(out1, [para_num, 1, 1])  # (batch_size * para_num, target_seq_len, d_model)

        attn2, attn_weights_l = self.mha2_l(enc_l, enc_l, out1r, padding_mask_l)
        # attn2 = tf.reshape(attn2, shape=(shape[0], shape[1], -1, shape[-1]))
        # attn_weights_l = tf.reshape(attn_weights_l, shape=(shape[0], shape[1], para_num, -1))
        # attn2.shape = (batch_size, tar_seq_len, para_num, d_model)
        # attn_weights_l.shape==(batch_size, tar_seq_len, para_num, inp_seq_len)

        pos_encoding = positional_encoding(para_num, self.d_model)
        attn2 += pos_encoding[:, tf.newaxis, :, :]

        attn3, attn_weights_g = self.mha2_g(attn2, attn2, out1r, padding_mask_g)
        attn3 = tf.reshape(attn3, shape=(shape[0], shape[1], -1, shape[-1]))
        # attn3.shape = (batch_size, tar_seq_len, para_num, d_model)

        attn3, attn_weights_g = self.mha2_g(attn2[:, 0, :, :], attn2[:, 0, :, :], out1[:, 0, :][:, tf.newaxis, :],
                                            padding_mask_g)
        # attn3.shape == (batch_size, 1, d_model)
        # attn_weights_g.shape == (batch_size, 1, para_num)

        for i in range(1, int(shape[1])):
            temp, temp1 = self.mha2_g(attn2[:, i, :, :], attn2[:, i, :, :], out1[:, i, :][:, tf.newaxis, :],
                                               padding_mask_g)
            attn3 = tf.concat((attn3, temp), axis=1)
            attn_weights_g = tf.concat((attn_weights_g, temp1), axis=1)

        attn3 = self.dropout2(attn3, training=training)
        out2 = self.layernorm2(attn3 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_g


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, w_emb, rate):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = w_emb
        self.pos_encoding = positional_encoding(target_vocab_size, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, para_num, enc_l, training, ranks, padding_mask_l):
        padding_mask_g = create_padding_mask(ranks)
        look_ahead_mask = create_look_ahead_mask(tf.shape(x)[1])
        dec_target_padding_mask = create_padding_mask(x)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        seq_len = tf.shape(x)[1]
        para_weights = 0

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, pw = self.dec_layers[i](x, para_num, enc_l, training,
                                                   combined_mask, padding_mask_g, padding_mask_l)
            para_weights += pw

        # x.shape == (batch_size, target_seq_len, d_model)
        # words_weights.shape == (batch_size, tar_seq_len, para_num, inp_seq_len)
        # para_weights.shape == (batch_size, tar_seq_len, para_num)
        return x,  para_weights


class MyModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, rate):
        super(MyModel, self).__init__()

        self.num_layers = num_layers

        self.vocab_size = vocab_size

        w_emb = tf.keras.layers.Embedding(vocab_size, d_model, trainable=True)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, vocab_size, w_emb, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size, w_emb, rate)

        self.out_layer = tf.keras.layers.Dense(vocab_size, activation=tf.nn.softmax)

    def call(self, inp, training, ranks, tar_inp, tar_real=None, cal_pw=False):
        con_words, padding_mask_l, para_num = self.encoder(inp, training, ranks)

        decoder_out, para_weights = self.decoder(tar_inp, para_num,
                                                            con_words, training, ranks, padding_mask_l)
        # para_weights = para_weights[1]

        pw = None
        if cal_pw:
            if tar_real is None:
                pw = tf.reduce_sum(para_weights, axis=1)
            else:
                tar_mask = tf.math.logical_not(tf.math.equal(tar_real, 0))
                tar_mask = tf.cast(tar_mask, dtype=para_weights.dtype)
                pw = tf.reduce_sum(tf.expand_dims(tar_mask, axis=-1) * para_weights, axis=1)

        vocab_dists = self.out_layer(decoder_out)  # (batch_size, target_seq_len, vocab_size)

        return vocab_dists, pw


if __name__ == "__main__":
    inp = tf.ones([32, 100, 256])
    ranks = tf.ones([32, 100])
    tar_inp = tf.ones([32, 512])
    sp = tf.ones([32, 100])
    encoded_sen_x = tf.ones([32, 100, 256])
    model =MyModel(2, 256, 4, 1028, 30000, 40000, 100, 0.5)
    a = model(inp, True, ranks, tar_inp,sp,encoded_sen_x)
    print(a)

