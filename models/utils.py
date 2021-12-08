
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.
    
    Args:
        - data_x: original data
        - data_x_hat: generated data
        - data_t: original time
        - data_t_hat: generated time
        - train_rate: ratio of training data from the original data
    """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate) :]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate) :]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
        - data: original data

    Returns:
        - time: extracted time information
        - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))

    return time, max_seq_len


def rnn_cell(module_name, hidden_dim):
    """Basic RNN Cell.

    Args:
        - module_name: gru, lstm, or lstmLN

    Returns:
        - rnn_cell: RNN Cell
    """
    assert module_name in ["gru", "lstm", "lstmLN"]

    # GRU
    if module_name == "gru":
        rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
    # LSTM
    elif module_name == "lstm":
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
    # LSTM Layer Normalization
    elif module_name == "lstmLN":
        rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
    return rnn_cell


def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    """Random vector generation.

    Args:
        - batch_size: size of the random vector
        - z_dim: dimension of random vector
        - T_mb: time information for the random vector
        - max_seq_len: maximum sequence length

    Returns:
        - Z_mb: generated random vector
    """
    Z_mb = list()
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim])
        temp_Z = np.random.uniform(0.0, 1, [T_mb[i], z_dim])
        temp[: T_mb[i], :] = temp_Z
        Z_mb.append(temp_Z)
    return Z_mb


def batch_generator(data, time, batch_size):
    """Mini-batch generator.

    Args:
        - data: time-series data
        - time: time information
        - batch_size: the number of samples in each batch

    Returns:
        - X_mb: time-series data in each batch
        - T_mb: time information in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb


#--------------------------------------------------------------


def get_angles(pos, i, d_model):
    """Get angle based on position.
        
        Args:
        - pos: data position
        - i: dimension
        - d_model: model dimension parameter
        
        Returns:
        - : angle
        """
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates



def positional_encoding(position, d_model):
    """Encode the positions.
        
        Args:
        - position: all positions up to this position are encoded
        - d_model: model dimension parameter
        
        Returns:
        - : encoded position
        """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
        
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
                                
    pos_encoding = angle_rads[np.newaxis, ...]
                                
    return tf.cast(pos_encoding, dtype=tf.float32)


class TokenAndPositionEmbedding(layers.Layer):
    """ Layer to encode the position information and add it to the data
        """
    def __init__(self, maxlen, embed_dim, ff_dim, b):
        """Initiate the layer
            
            Args:
            - maxlen: all positions up to this position are encoded
            - embed_dim: embedding dimension
            - ff_dim: feed forward network hidden layer dimension
            - b: if b is true, the data dimension remains unchanged and no feed forward networks are applied
            """
        super(TokenAndPositionEmbedding, self).__init__()
        if b:
            self.ffn = keras.Sequential(
                                        [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
                                        )
        self.pos_emb = positional_encoding(maxlen, embed_dim)
        self.b = b
    
    def call(self, x):
        """embed the position information and add it to the data
            
            Args:
            - x: input time-series data
            
            Returns:
            - : position information added to the data
            """
        seq_len = tf.shape(x)[1]
        if self.b:
            x = self.ffn(x)
        return x + self.pos_emb[:, :seq_len, :]



def create_padding_mask(seq):
    """unused
        """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    """creates look ahead mask for the transformer decoder. When predicting data at position i, it masks the data at future steps.
        
        Args:
        - size: time-series data size
        
        Returns:
        - mask: look ahead mask
        """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
        
    Args:
    - q: query shape == (..., seq_len_q, depth)
    - k: key shape == (..., seq_len_k, depth)
    - v: value shape == (..., seq_len_v, depth_v)
    - mask: Float tensor with shape broadcastable
    to (..., seq_len_q, seq_len_k). Defaults to None.
        
    Returns:
    - output, attention_weights
    """
            
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
                        
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
                        
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
                            
    return output


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention layer
        """
    def __init__(self, d_model, num_heads):
        """initialize the layer.
            
            Args:
            - d_model: model dimension parameter
            - num_heads: number of splitting heads
            
            """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
    
        assert d_model % self.num_heads == 0
    
        self.depth = d_model // self.num_heads
        
        # linear projection layers to d_model dimension
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
    
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """splits the data into self.num_heads parts
            
            Args:
            - x: time-series data
            - batch_size: batch size
            
            Returns:
            - : splitted data
            """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        """Apply multi-head attention layer on the data.
            
            Args:
            - q: query
            - k: key
            - v: value
            - mask: applied mask on the data
            
            Returns:
            - output: output time-series data
            """
        
        batch_size = tf.shape(q)[0]
    
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
    
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention = scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
                                                                       
        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
                                                                       
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
                                                                       
        return output


def point_wise_feed_forward_network(d_model, dff):
    """ two-layer feed forward network
        """
    return tf.keras.Sequential([
                                tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
                                tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
                                ])

class EncoderLayer(tf.keras.layers.Layer):
    """Encoder layer of the transformer network
        """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """initialize the layer.
            
            Args:
            - d_model: model dimension parameter
            - num_heads: number of splitting heads
            - dff: the hidden layer dimension of the feed forward network
            - rate: dropout rate

            """
        super(EncoderLayer, self).__init__()
        
        # multi-head attention module
        self.mha = MultiHeadAttention(d_model, num_heads)
        # two layer feed forward network
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        # layer normalization layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # dropout layers
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):
        """Encode the time-series data into the latent space. At first a self-attention module is applied followed by a dropout layer and a layer-normalization layer. Then a feed forward network is applied followed by the same two networks.
            
            Args:
            - x: time-series data
            - training: if true, we are at the training stage
            - mask: the mask applied on the data
            
            Returns:
            - out2: encoded data
            """

        attn_output = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class DecoderLayer(tf.keras.layers.Layer):
    """Decoder layer of the transformer network
        """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """initialize the layer.
            
            Args:
            - d_model: model dimension parameter
            - num_heads: number of splitting heads
            - dff: the hidden layer dimension of the feed forward network
            - rate: dropout rate
            
            """
        super(DecoderLayer, self).__init__()
        
        # multi-head attention modules
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        # two layer feed forward network
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        # layer normalization layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # dropout layers
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        
        
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """Decode the encoded data. At first a self-attention module is applied followed by a dropout layer and a layer-normalization layer. The input to this layer is masked by a look ahead mask. Then, a multi-head attention module is applied (with the encoded data as keys and values, and the time-series data as input) followed by a dropout layer and a layer-normalization layer. Then a feed forward network is applied followed by the same two networks.
            
            Args:
            - x: time-series data
            - training: if true, we are at the training stage
            - enc_output: the encoded data
            - look_ahead_mask: the look ahead mask applied on the data
            - padding_mask: the padding mask applied on the data
            
            Returns:
            - out3: decoded time-series data
            """

        attn1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3


def CustomSchedule(step, d_model, warmup_steps=2000):
    # unused
    d_model = float(d_model)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (warmup_steps ** -1.5)
        
    return tf.math.rsqrt(d_model) * tf.math.minimum(arg1, arg2) 
