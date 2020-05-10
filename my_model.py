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
                 w_emb, rate, mode):
        super(Encoder, self).__init__()

        self.para_encoder = SenEncoder(word_enc_layer, d_model, num_heads, dff, input_vocab_size, w_emb, rate, mode)

        self.d_model = d_model
        self.mode = mode

    def call(self, inp, training, ranks):
        shape = inp.shape
        para_num = int(shape[1])

        inp = tf.reshape(inp, shape=[-1, shape[-1]])  # shape == (batch_size * para_num, inp_seq_len)
        padding_mask_l = create_padding_mask(inp)

        padding_mask = create_padding_mask(inp)
        output_mask = create_output_mask(inp)
        if self.mode == 'v':
            # (batch_size * para_num, d_model), (batch_size * para_num, inp_seq_len, d_model)
            con_words, _ = self.para_encoder(inp, training, padding_mask, output_mask)

            return con_words, padding_mask_l, para_num
        else:
            pos_encoding = positional_encoding(para_num, self.d_model)

            con_words, para_encoder = self.para_encoder(inp, training, padding_mask, output_mask)

            para_encoder = tf.reshape(para_encoder, [-1, para_num, self.d_model])  # (batch_size, para_num, d_model)
            # para_encoder += self.rank_embedding(ranks)
            para_encoder += pos_encoding

            return con_words, padding_mask_l, para_encoder


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate, mode):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2_g = MultiHeadAttention(d_model, num_heads)
        self.mha2_l = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        self.dropout4 = tf.keras.layers.Dropout(rate)

        self.d_model = d_model
        self.mode = mode
        self.pos_encoding = positional_encoding(100, self.d_model)

    def call(self, x, para_num, enc_l, training, look_ahead_mask, padding_mask_g, padding_mask_l):
        """
        :param enc_l: shape == (batch_size * para_num, inp_seq_len, d_model)

        """
        if self.mode == 'v':
            attn1, _ = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
            attn1 = self.dropout1(attn1, training=training)
            out1 = self.layernorm1(attn1 + x)
            shape = tf.shape(attn1)

            out1r = tf.tile(out1, [1, para_num, 1])  # (batch_size * para_num, target_seq_len, d_model)
            out1r = tf.reshape(out1r, (-1, shape[1], shape[-1]))

            attn2, attn_weights_l = self.mha2_l(enc_l, enc_l, out1r, padding_mask_l)
            attn2 = tf.reshape(attn2, shape=(shape[0], -1, shape[1], shape[-1]))
            attn2 = tf.transpose(attn2, [0, 2, 1, 3])
            # attn2.shape = (batch_size, tar_seq_len, para_num, d_model)
            # attn_weights_l = tf.reshape(attn_weights_l, shape=(shape[0], shape[1], para_num, -1))
            # attn_weights_l.shape==(batch_size, tar_seq_len, para_num, inp_seq_len)

            # attn2 *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            attn2 += self.pos_encoding[:, tf.newaxis, :para_num, :]

            attn2 = self.dropout4(attn2, training=training)
            attn2 = self.layernorm4(attn2)

            attn2x = tf.reshape(attn2, shape=(-1, para_num, self.d_model))
            out1c = tf.reshape(out1, shape=(-1, 1, self.d_model))

            attn3, attn_weights_g = self.mha2_g(attn2x, attn2x, out1c, padding_mask_g)
            attn3 = tf.reshape(attn3, shape=(shape[0], shape[1], self.d_model))
            attn_weights_g = tf.reshape(attn_weights_g, shape=(shape[0], -1, para_num))
            # print(attn_weights_g[0,:,:])

            attn3 = self.dropout2(attn3, training=training)
            out2 = self.layernorm2(attn3 + out1)  # (batch_size, target_seq_len, d_model)

            ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
            ffn_output = self.dropout3(ffn_output, training=training)
            out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

            return out3, attn_weights_g
        else:
            enc_g = para_num
            para_num = tf.shape(enc_g)[1]
            attn1, _ = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
            attn1 = self.dropout1(attn1, training=training)
            out1 = self.layernorm1(attn1 + x)

            attn_g, attn_weights_g = self.mha2_g(
                enc_g, enc_g, out1, padding_mask_g)  # attn_weights_g.shape == (batch_size, tar_seq_len, para_num)

            out1r = tf.tile(out1, [1, para_num, 1])  # (batch_size * para_num, target_seq_len, d_model)
            shape = tf.shape(attn1)
            out1r = tf.reshape(out1r, (-1, shape[1], shape[-1]))

            attn_weights_gex = tf.expand_dims(attn_weights_g, -1)
            attn2, attn_weights_l = self.mha2_l(enc_l, enc_l, out1r, padding_mask_l)

            # attn2.shape = (batch_size, tar_seq_len, para_num, d_model)
            # attn_weights_l.shape==(batch_size, tar_seq_len, para_num, inp_seq_len)
            attn2 = tf.reshape(attn2, shape=(shape[0], -1, shape[1], shape[-1]))
            attn2 = tf.transpose(attn2, [0, 2, 1, 3])
            attn_weights_l = tf.reshape(attn_weights_l, shape=(shape[0], para_num, shape[1], -1))
            attn_weights_l = tf.transpose(attn_weights_l, [0, 2, 1, 3])
            attn2 = tf.reduce_sum(tf.multiply(attn2, attn_weights_gex), axis=-2)  # (batch_size, tar_seq_len, d_model)

            attn2 = self.dropout2(attn2 + attn_g, training=training)
            out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
            words_weights = tf.multiply(attn_weights_l,
                                        attn_weights_gex)  # (batch_size, tar_seq_len, para_num, inp_seq_len)

            ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
            ffn_output = self.dropout3(ffn_output, training=training)
            out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

            return out3, attn_weights_g


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, w_emb, rate, mode):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = w_emb
        self.pos_encoding = positional_encoding(target_vocab_size, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate, mode)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        self.mode = mode

    def call(self, x, para_num, enc_l, training, ranks, padding_mask_l):
        if self.mode == 'v':
            tar_seq_len = tf.shape(x)[1]
            ranks = tf.tile(ranks, [1, tar_seq_len])
            ranks = tf.reshape(ranks, (-1, para_num))

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
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, rate, mode):
        super(MyModel, self).__init__()

        self.num_layers = num_layers

        self.vocab_size = vocab_size

        w_emb = tf.keras.layers.Embedding(vocab_size, d_model, trainable=True)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, vocab_size, w_emb, rate, mode)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size, w_emb, rate, mode)

        self.out_layer = tf.keras.layers.Dense(vocab_size, activation=tf.nn.softmax)

    def call(self, inp, training, ranks, tar_inp, tar_real=None, cal_pw=False):
        con_words, padding_mask_l, para_num = self.encoder(inp, training, ranks)

        decoder_out, para_weights = self.decoder(tar_inp, para_num,
                                                            con_words, training, ranks, padding_mask_l)
        # para_weights = para_weights[0]

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



