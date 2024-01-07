import tensorflow as tf
import tensorflow.keras.backend as K


class MogrifierLayer(tf.keras.layers.Layer):

    def __init__(self, dimWeightMatrix=32, dimHiddenState=32, dimQK=32, numMogrifyRounds=5, units=32,
                 state_size=tf.TensorShape([128, 128]), **kwargs):
        super(MogrifierLayer, self).__init__()
        self.units = units
        self.numRounds = numMogrifyRounds
        self.state_size = [state_size, state_size]
        self.dimWeight = dimWeightMatrix
        self.dimHidden = dimHiddenState
        self.dimQK = dimQK

    # Adding config definition for usage in tf.keras.layers.Bidirectional
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'numRounds': self.numRounds,
            'state_size': self.state_size,
            'dimWeight': self.dimWeight,
            'dimHidden': self.dimHidden,
            'dimQK': self.dimQK
        })
        return config

    def build(self, input_shape):

        batch_size = 1

        embedding = input_shape[1]
        print(input_shape[-1],input_shape[0], input_shape[1])

        print(f"batch_size: {batch_size}, embedding: {embedding}")

        self.list_wx = []
        self.list_wh = []
        self.list_b = []
        self.list_dense_x = []
        self.list_dense_h = []

        for i in range(4):
            self.list_wx.append(self.add_weight(name='wx'+ str(i), shape=(batch_size, self.dimWeight),
                                                initializer='glorot_uniform',
                                                trainable=True))
            self.list_wh.append(self.add_weight(name='wh' + str(i), shape=(batch_size, self.dimWeight),
                                                initializer='glorot_uniform',
                                                trainable=True))
            self.list_b.append(self.add_weight(name='b' + str(i), shape=(embedding,),
                                               initializer='glorot_uniform',
                                               trainable=True))
            self.list_dense_x.append(tf.keras.layers.Dense(batch_size))
            self.list_dense_h.append(tf.keras.layers.Dense(batch_size))

        self.h_state = self.add_weight(name='h_state', shape=(batch_size, self.dimHidden),
                                       initializer='glorot_uniform',
                                       trainable=True)
        self.hiddenDenseContract = tf.keras.layers.Dense(embedding)
        self.hiddenDenseExpand = tf.keras.layers.Dense(self.dimHidden)

        self.c_state = self.add_weight(name='c_state', shape=(batch_size, self.dimHidden),
                                       initializer='glorot_uniform',
                                       trainable=True)
        self.cellDenseContract = tf.keras.layers.Dense(embedding)
        self.cellDenseExpand = tf.keras.layers.Dense(self.dimHidden)

        # Building layers to be used for Mogrification (The Q and K matrices)
        self.qk_list = []
        self.qk_denseContract_list = []
        self.qk_denseExpand_list = []

        for i in range(self.numRounds):
            self.qk_list.append(
                self.add_weight(name='qk' + str(i), shape=(batch_size, self.dimQK),
                                initializer='glorot_uniform',
                                trainable=True)
            )

            self.qk_denseContract_list.append(
                tf.keras.layers.Dense(batch_size)
            )

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [self.h_state, self.c_state]

    # Mogrification function
    def mogrify(self, x, h, rounds):
        for i in range(rounds):
            if (i % 2 == 0):
                h = self.hiddenDenseExpand(
                    tf.math.multiply(2 * tf.math.sigmoid(tf.matmul(self.qk_denseContract_list[i](self.qk_list[i]), x)),
                                     self.hiddenDenseContract(h)))
            else:
                x = tf.math.multiply(2 * tf.math.sigmoid(
                    tf.matmul(self.qk_denseContract_list[i](self.qk_list[i]), self.hiddenDenseContract(h))), x)
        return x, h

    def call(self, inputs, states):

        # Find values after mogrification
        x_val, h_val = self.mogrify(inputs, states[0], self.numRounds)

        # Standard LSTM update equations
        f = tf.math.sigmoid(
            tf.matmul(self.list_dense_x[0](self.list_wx[0]), x_val) + tf.matmul(self.list_dense_h[0](self.list_wh[0]),
                                                                                self.hiddenDenseContract(h_val)) +
            self.list_b[0])
        i = tf.math.sigmoid(
            tf.matmul(self.list_dense_x[1](self.list_wx[1]), x_val) + tf.matmul(self.list_dense_h[1](self.list_wh[1]),
                                                                                self.hiddenDenseContract(h_val)) +
            self.list_b[1])
        j = tf.math.tanh(
            tf.matmul(self.list_dense_x[2](self.list_wx[2]), x_val) + tf.matmul(self.list_dense_h[2](self.list_wh[2]),
                                                                                self.hiddenDenseContract(h_val)) +
            self.list_b[2])
        o = tf.math.sigmoid(
            tf.matmul(self.list_dense_x[3](self.list_wx[3]), x_val) + tf.matmul(self.list_dense_h[3](self.list_wh[3]),
                                                                                self.hiddenDenseContract(h_val)) +
            self.list_b[3])

        # Update Cell State and Hidden State
        c = tf.math.multiply(f, self.cellDenseContract(states[1])) + tf.math.multiply(i, j)
        h = tf.math.multiply(o, tf.math.tanh(c))

        self.h_state = self.hiddenDenseExpand(h)
        self.c_state = self.cellDenseExpand(c)

        return o, (self.h_state, self.c_state)


class MogrifierLSTM(tf.keras.layers.Layer):
    '''
    This is a simple wrapper function for making a MogrifierLSTM Layer directly instead of
    having to custom define using tf.keras.layers.RNN
    '''

    def __init__(self, dimWeightMatrix=32, dimHiddenState=32, dimQK=32, numMogrifyRounds=5, units=32,
                 state_size=tf.TensorShape([128, 128]), return_sequences=True, return_state=False, go_backwards=False,
                 **kwargs):
        super(MogrifierLSTM, self).__init__()
        self.lstmCell = MogrifierLayer(dimWeightMatrix=dimWeightMatrix, dimHiddenState=dimHiddenState, dimQK=dimQK,
                                       numMogrifyRounds=numMogrifyRounds, units=units, state_size=state_size)
        self.go_backwards = go_backwards
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.lstm = tf.keras.layers.RNN(self.lstmCell, return_sequences=self.return_sequences,
                                        return_state=return_state, go_backwards=self.go_backwards)

    # Adding config definition for usage in tf.keras.layers.Bidirectional
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'lstmCell': self.lstmCell,
            'lstm': self.lstm,
            'go_backwards': self.go_backwards,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state
        })

        return config

    def call(self, input):
        y = self.lstm(input)
        return y

# class ParaphraseDetector(tf.keras.Model):
#   def __init__(self, vocab_size = 100000, textVec1 = None, textVec2 = None, embed_dim = 300, dimWeightMatrix = 16, dimHiddenState = 16, dimQK = 16, numMogrifyRounds = 5, units=32, return_sequences = False, state_size = tf.TensorShape([None, None])):
#
#     super(ParaphraseDetector, self).__init__()
#
#     self.textVec1 = textVec1
#     self.embedding1 = tf.keras.layers.Embedding(vocab_size, embed_dim)
#     self.forward_layer1 = MogrifierLSTM(dimWeightMatrix = dimWeightMatrix, dimHiddenState = dimHiddenState, dimQK = dimQK, numMogrifyRounds = numMogrifyRounds, units = units, return_sequences = return_sequences, state_size = state_size)
#     self.backward_layer1 = MogrifierLSTM(dimWeightMatrix = dimWeightMatrix, dimHiddenState = dimHiddenState, dimQK = dimQK, numMogrifyRounds = numMogrifyRounds, units = units, return_sequences = return_sequences, go_backwards = True, state_size = state_size)
#     self.bidirectional1 = tf.keras.layers.Bidirectional(layer = self.forward_layer1, backward_layer = self.backward_layer1, merge_mode = 'sum')
#
#     # self.textVec2 = textVec2
#     # self.embedding2 = tf.keras.layers.Embedding(vocab_size, embed_dim)
#     # self.forward_layer2 = MogrifierLSTM(dimWeightMatrix = dimWeightMatrix, dimHiddenState = dimHiddenState, dimQK = dimQK, numMogrifyRounds = numMogrifyRounds, units = units, return_sequences = return_sequences, state_size = state_size)
#     # self.backward_layer2 = MogrifierLSTM(dimWeightMatrix = dimWeightMatrix, dimHiddenState = dimHiddenState, dimQK = dimQK, numMogrifyRounds = numMogrifyRounds, units = units, return_sequences = return_sequences, go_backwards = True, state_size = state_size)
#     # self.bidirectional2 = tf.keras.layers.Bidirectional(layer = self.forward_layer2, backward_layer = self.backward_layer2, merge_mode = 'sum')
#
#     self.flat = tf.keras.layers.Flatten()
#     self.concat = tf.keras.layers.Concatenate()
#     self.dense1 = tf.keras.layers.Dense(100, activation = 'sigmoid', kernel_initializer= 'random_normal')
#
#     self.dense_output = tf.keras.layers.Dense(1, activation = 'softmax', kernel_initializer= 'random_normal')
#
#   def call(self, q1):
#
#     vec1 = self.textVec1(q1)
#     # vec2 = self.textVec1(q2)
#
#     embed1 = self.embedding1(vec1)
#     # embed2 = self.embedding2(vec2)
#
#     rnn1 = self.bidirectional1(embed1)
#     # rnn2 = self.bidirectional2(embed2)
#
#     flat1 = self.flat(rnn1)
#     # flat2 = self.flat(rnn2)
#
#     # concat = self.concat([flat1, flat2])
#
#     return self.dense_output(self.dense1(flat1))
