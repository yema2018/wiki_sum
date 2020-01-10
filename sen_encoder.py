from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    sines = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)

    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def rel_pos_encoding(seq_len):
    a = tf.range(seq_len, 0, -1) - tf.range(seq_len-1, -1, -1)[:, np.newaxis]
    b = tf.linalg.band_part(a, -1, 0)
    return b


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_output_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, :, tf.newaxis, tf.newaxis]


class AttLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_head, dff, rate):
        super(AttLayer, self).__init__()
        self.emb_size = d_model
        self.num_head = num_head
        self.weight = tf.Variable(initial_value=tf.random_normal_initializer()([d_model]), trainable=True)
        self.full_conn = tf.keras.layers.Dense(d_model)
        self.concat = tf.keras.layers.Dense(d_model)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

    def call(self, inputs, mask, training):
        h = self.full_conn(inputs)
        batch_size = tf.shape(h)[0]
        depth = self.emb_size // self.num_head

        h = tf.reshape(h, shape=[batch_size, -1, self.num_head, depth])
        self.weight = tf.reshape(self.weight, shape=[self.num_head, depth])
        self.weight = self.weight[tf.newaxis, tf.newaxis, :, :]

        logits = tf.reduce_sum(tf.multiply(self.weight, h), keepdims=True, axis=-1)
        logits /= tf.math.sqrt(tf.cast(depth, tf.float32))
        logits += (mask * -1e19)
        alpha = tf.nn.softmax(logits, axis=1, name='alpha')
        out = tf.reduce_sum(tf.multiply(h, alpha), axis=1)

        concat = tf.reshape(out, shape=[batch_size, -1])
        concat = self.concat(concat)  # shape==(batch, d_model)

        concat = self.dropout1(concat, training=training)
        concat = self.layernorm1(concat)

        ffn_output = self.ffn(concat)  # (batch_size, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out = self.layernorm2(concat + ffn_output)  # (batch_size, d_model)

        return out


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e19)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def scaled_dot_product_attention_2(q, k, v, relp, mask):
    """
    :param relp: relative position embedding, shape == (seq_len_v, seq_len_v, depth_v)
    seq_len_q == seq_len_k == seq_len_v
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e19)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    v1 = tf.stack([v for _ in range(q.shape[-2])], axis=-3)  # (..., seq_len_q, seq_len_v, depth_v)
    v1 += relp
    output = tf.reduce_sum(tf.multiply(v1, tf.expand_dims(attention_weights, -1)), axis=-2)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, no_relp=True):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.no_relp = no_relp

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask, relp=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        if self.no_relp:
            # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
            # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
            scaled_attention, attention_weights = scaled_dot_product_attention(
                q, k, v, mask)
        else:
            shape = relp.shape
            relp = tf.reshape(relp, [shape[0], shape[1], -1, self.num_heads])
            relp = tf.transpose(relp, perm=[3, 0, 1, 2])
            scaled_attention, attention_weights = scaled_dot_product_attention_2(
                q, k, v, relp, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, tf.reduce_mean(attention_weights, axis=1)


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class BaseEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, w_emb, rate):
        super(BaseEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = w_emb
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class SenEncoder(tf.keras.layers.Layer):
    """
    encode the sentences/paragraphs in a news cluster.
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, w_emb, rate, mode):
        super(SenEncoder, self).__init__()

        self.encoder = BaseEncoder(num_layers, d_model, num_heads, dff, input_vocab_size, w_emb, rate)
        self.mode = mode
        if mode == 'p':
            self.final_encoder = AttLayer(d_model, num_heads, dff, rate)

    def call(self, inp, training, padding_mask, output_mask):
        para_emb = None
        contextual_words = self.encoder(inp, training, padding_mask)  # (batch_size, inp_seq_len, d_model)

        if self.mode == 'p':
            para_emb = self.final_encoder(contextual_words, output_mask, training)  # (batch_size, d_model)

        return contextual_words, para_emb



if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    a=positional_encoding(25, 256)
    b = positional_encoding(100,256)
    print(a)
    print(b)
