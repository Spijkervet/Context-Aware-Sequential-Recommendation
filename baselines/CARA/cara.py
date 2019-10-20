from keras.engine import InputSpec
from keras.layers import Recurrent, initializations, activations, regularizers, time_distributed_dense, SimpleRNN, GRU, \
    LSTM
from keras import backend as K

import numpy as np
from keras.models import Sequential

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

class CARA(GRU):
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W = dropout_W
        self.dropout_U = dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(GRU, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = 10

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]

        self.W_z = self.add_weight((self.input_dim, self.output_dim),
                                   initializer=self.init,
                                   name='{}_W_z'.format(self.name),
                                   regularizer=self.W_regularizer)
        self.U_z = self.add_weight((self.output_dim, self.output_dim),
                                   initializer=self.init,
                                   name='{}_U_z'.format(self.name),
                                   regularizer=self.W_regularizer)
        self.b_z = self.add_weight((self.output_dim,),
                                   initializer='zero',
                                   name='{}_b_z'.format(self.name),
                                   regularizer=self.b_regularizer)
        self.W_r = self.add_weight((self.input_dim, self.output_dim),
                                   initializer=self.init,
                                   name='{}_W_r'.format(self.name),
                                   regularizer=self.W_regularizer)
        self.U_r = self.add_weight((self.output_dim, self.output_dim),
                                   initializer=self.init,
                                   name='{}_U_r'.format(self.name),
                                   regularizer=self.W_regularizer)
        self.b_r = self.add_weight((self.output_dim,),
                                   initializer='zero',
                                   name='{}_b_r'.format(self.name),
                                   regularizer=self.b_regularizer)
        self.W_h = self.add_weight((self.input_dim, self.output_dim),
                                   initializer=self.init,
                                   name='{}_W_h'.format(self.name),
                                   regularizer=self.W_regularizer)
        self.U_h = self.add_weight((self.output_dim, self.output_dim),
                                   initializer=self.init,
                                   name='{}_U_h'.format(self.name),
                                   regularizer=self.W_regularizer)
        self.b_h = self.add_weight((self.output_dim,),
                                   initializer='zero',
                                   name='{}_b_h'.format(self.name),
                                   regularizer=self.b_regularizer)

        self.A_h = self.add_weight((self.output_dim, self.output_dim),
                                   initializer=self.init,
                                   name='{}_A_h'.format(self.name),
                                   regularizer=self.W_regularizer)
        self.A_u = self.add_weight((self.output_dim, self.output_dim),
                                   initializer=self.init,
                                   name='{}_A_u'.format(self.name),
                                   regularizer=self.W_regularizer)

        self.b_a_h = self.add_weight((self.output_dim,),
                                     initializer='zero',
                                     name='{}_b_a_h'.format(self.name),
                                     regularizer=self.b_regularizer)
        self.b_a_u = self.add_weight((self.output_dim,),
                                     initializer='zero',
                                     name='{}_b_a_u'.format(self.name),
                                     regularizer=self.b_regularizer)


        self.W_t = self.add_weight((self.input_dim, self.output_dim),
                                   initializer=self.init,
                                   name='{}_W_t'.format(self.name),
                                   regularizer=self.W_regularizer)
        self.U_t = self.add_weight((1, self.output_dim),
                                   initializer=self.init,
                                   name='{}_U_t'.format(self.name),
                                   regularizer=self.W_regularizer)
        self.b_t = self.add_weight((self.output_dim,),
                                   initializer='zero',
                                   name='{}_b_t'.format(self.name),
                                   regularizer=self.b_regularizer)

        self.W_g = self.add_weight((self.input_dim, self.output_dim),
                                   initializer=self.init,
                                   name='{}_W_g'.format(self.name),
                                   regularizer=self.W_regularizer)
        self.U_g = self.add_weight((1, self.output_dim),
                                   initializer=self.init,
                                   name='{}_U_g'.format(self.name),
                                   regularizer=self.W_regularizer)
        self.b_g = self.add_weight((self.output_dim,),
                                   initializer='zero',
                                   name='{}_b_g'.format(self.name),
                                   regularizer=self.b_regularizer)



        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def preprocess_input(self, x):
        return x

    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        B_U = states[1]  # dropout matrices for recurrent units
        B_W = states[2]

        u = x[:, self.output_dim: 2 * self.output_dim]
        t = x[:, 2 * self.output_dim: (2 * self.output_dim) + 1]
        g = x[:, (2 * self.output_dim) + 1:]
        x = x[:, :self.output_dim]
        
        t = self.inner_activation(K.dot(t, self.U_t))
        g = self.inner_activation(K.dot(g, self.U_g))
#       Time-based gate
        T = self.inner_activation(K.dot(x, self.W_t) + t + self.b_t)
#       Geo-based gate
        G = self.inner_activation(K.dot(x, self.W_g) + g + self.b_g)

#       Contextual Attention Gate
        a = self.inner_activation(
            K.dot(h_tm1, self.A_h) + K.dot(u, self.A_u) + self.b_a_h + self.b_a_u)

        x_z = K.dot(x, self.W_z) + self.b_z
        x_r = K.dot(x, self.W_r) + self.b_r
        x_h = K.dot(x, self.W_h) + self.b_h

        u_z_ = K.dot((1 - a) * u, self.W_z) + self.b_z
        u_r_ = K.dot((1 - a) * u, self.W_r) + self.b_r
        u_h_ = K.dot((1 - a) * u, self.W_h) + self.b_h

        u_z = K.dot(a * u, self.W_z) + self.b_z
        u_r = K.dot(a * u, self.W_r) + self.b_r
        u_h = K.dot(a * u, self.W_h) + self.b_h

#       update gate
        z = self.inner_activation(x_z + K.dot(h_tm1, self.U_z) + u_z)
#       reset gate
        r = self.inner_activation(x_r + K.dot(h_tm1, self.U_r) + u_r)
#       hidden state
        hh = self.activation(x_h + K.dot(r * T * G * h_tm1, self.U_h) + u_h)

        h = z * h_tm1 + (1 - z) * hh
        h = (1 + u_z_ + u_r_ + u_h_) * h
        return h, [h]


import numpy as np
from keras.models import Model, Sequential
from keras.layers import Embedding, Input, merge, SimpleRNN, Activation, Dense, Flatten, GlobalAveragePooling1D, GRU, \
    LSTM
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
import itertools
from keras.regularizers import l2

def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

def bpr_triplet_loss(X):
    positive_item_latent, negative_item_latent = X

    reg = 0

    loss = 1 - K.log(K.sigmoid(
        K.sum(positive_item_latent, axis=-1, keepdims=True) -
        K.sum(negative_item_latent, axis=-1, keepdims=True))) - reg

    return loss

# Context-Aware Venue Recommendation with pairwise ranking function
class Recommender():
    def __init__(self, num_users, num_items, num_times, latent_dim, maxVenue):

        self.maxVenue = maxVenue
        self.latent_dim = latent_dim
        
#       Inputs
        self.user_input = Input(shape=(1,), dtype='int32', name='user_input')
        self.checkins_input = Input(shape=(self.maxVenue,), dtype='int32', name='venue_input')
        self.neg_checkins_input = Input(shape=(self.maxVenue,), dtype='int32', name='neg_venue_input')
        self.time_input = Input(shape=(self.maxVenue,), dtype='int32', name='time_input')
        self.gap_time_input = Input(shape=(self.maxVenue, 1,), dtype='float32', name='time_interval_input')
        
        self.u_embedding = Embedding(input_dim=num_users, output_dim=latent_dim, name='user_embedding', 
                                     init=init_normal)
        self.v_embedding = Embedding(input_dim=num_items, output_dim=latent_dim, name='venue_embedding',
                                     init=init_normal) 
        self.t_embedding = Embedding(input_dim=num_times, output_dim=latent_dim, name='time_embedding',
                                     init=init_normal) 


#       User latent factor
        self.u_latent = Flatten()(self.u_embedding(self.user_input))
        self.t_latent = Flatten()(self.t_embedding(self.time_input))
       
        rnn_input = merge(
                [self.v_embedding(self.checkins_input), self.t_embedding(self.time_input), self.gap_time_input],
                mode="concat")
        neg_rnn_input = merge(
                [self.v_embedding(self.neg_checkins_input), self.t_embedding(self.time_input), self.gap_time_input],
                mode="concat")


#         rnn_input = self.v_embedding(self.checkins_input)
#         neg_rnn_input = self.v_embedding(self.neg_checkins_input)
        
        
        self.pos_distance_input = Input(shape=(self.maxVenue, 1,), dtype='float32', name='pos_distance_input')
        self.neg_distance_input = Input(shape=(self.maxVenue, 1,), dtype='float32', name='neg_distance_input')
        rnn_input = merge([rnn_input, self.pos_distance_input], mode="concat")
        neg_rnn_input = merge([neg_rnn_input, self.neg_distance_input], mode="concat")


        self.rnn = Sequential()
#       latent_dim * 2 + 2 = v_embedding + t_embedding + time_gap + distance

        self.rnn.add(
                        CARA(latent_dim, input_shape=(self.maxVenue, (self.latent_dim * 2) + 2,), unroll=True))
        

        self.checkins_emb = self.rnn(rnn_input)
        self.neg_checkins_emb = self.rnn(neg_rnn_input)

        pred = merge([self.checkins_emb, self.u_latent], mode="dot")
        neg_pred = merge([self.neg_checkins_emb, self.u_latent], mode="dot")

        
        INPUT = [self.user_input, self.time_input, self.gap_time_input, self.pos_distance_input,
                 self.neg_distance_input, self.checkins_input,
                 self.neg_checkins_input]

        loss = merge([pred, neg_pred], mode=bpr_triplet_loss, name='loss', output_shape=(1,))
        self.model = Model(input=INPUT, output=loss)
        self.model.compile(optimizer=Adam(), loss=identity_loss)
        
    

    def rank(self, uid, hist_venues, hist_times, hist_time_gap, hist_distances):
        
#         hist_venues = hist_venues + [candidate_venue]
#         hist_times = hist_times + [time]
#         hist_time_gap = hist_time_gap + [time_gap]
#         hist_distances = hist_distances + [distance]

        u_latent = self.model.get_layer('user_embedding').get_weights()[0][uid]
        v_latent = self.model.get_layer('venue_embedding').get_weights()[0][hist_venues]
        t_latent = self.model.get_layer('time_embedding').get_weights()[0][hist_times]
        rnn_input = np.concatenate([t_latent, hist_time_gap], axis=-1)
        rnn_input = np.concatenate([rnn_input, hist_distances], axis=-1)

        rnn_input = np.concatenate([v_latent, rnn_input], axis=-1)

        dynamic_latent = self.rnn.predict(rnn_input)

        scores = np.dot(dynamic_latent, u_latent)
        return scores


uNum = 10
vNum = 10
tNum = 10
num_instances = 10
maxVenue = 5
randomeContinuousValue = 100


rec = Recommender(10,10,10,10,5)


users = np.random.randint(uNum, size=(num_instances))
times = np.random.randint(uNum, size=(num_instances, maxVenue))
time_gaps = np.random.randint(randomeContinuousValue, size=(num_instances, maxVenue, 1))
# random distance for visited venues
pos_distances = np.random.randint(randomeContinuousValue, size=(num_instances, maxVenue, 1))
neg_distances = np.random.randint(randomeContinuousValue, size=(num_instances, maxVenue, 1))
checkins = np.random.randint(vNum, size=(num_instances, maxVenue))
neg_checkins = np.random.randint(vNum, size=(num_instances, maxVenue))

X = [users, times, time_gaps, pos_distances, neg_distances, checkins, neg_checkins]
y = np.array([1]*num_instances)


from random import shuffle, choice
sequences = []
allvenues = set()
with open('../../data/ml-1m.txt', 'r') as f:
    lastuser = None
    for line in f:
        line = [int(x) for x in line.split()]
        allvenues.add(line[1])
        
        if lastuser == None:
            lastuser = line[0]
            seq = (line[0], [])
        elif line[0] != lastuser:
            sequences.append(seq)
            lastuser = line[0]
            seq = (line[0], [])
        else:
            seq[1].append((line[1], line[2]))
allvenues = list(allvenues)


len(allvenues)
print(len(sequences))


# IMPORTANT: uses global variables from previous cell
max_seq_len = 30
def batch_generator(batch_size=32, shuffle_every_epoch=True, status="train"):
    if status.lower() == 'train':
        end = -3
    elif status.lower() == 'validate':
        end = -2
    elif status.lower() == 'test':
        end = -1
    else:
        raise TypeError("Status keyword argument must be either 'train', 'test' or 'validate'.")

    if shuffle_every_epoch:
        shuffle(sequences)
                
    batch_pointer = 0
    while True:
        if batch_pointer + batch_size > len(sequences):
            batch_pointer = 0
            shuffle(sequences)

        batch_seqs = sequences[batch_pointer:batch_pointer+batch_size]
        batch_pointer += batch_size

        users = np.array([seq[0] for seq in batch_seqs])
        times = np.zeros((batch_size, max_seq_len))
        time_gaps = np.zeros((batch_size, max_seq_len, 1))
        for i, seq in enumerate(batch_seqs):
            actual_seq = seq[1]
            to_pad = len(actual_seq[:max_seq_len]) - max_seq_len
            for j, data in enumerate(actual_seq[:max_seq_len]):                
                if not j:
                    continue
                time_gaps[i, j+to_pad, 0] = data[1] - actual_seq[j-1][1]

        pos_distances = np.zeros((batch_size, max_seq_len, 1))
        neg_distances = np.zeros((batch_size, max_seq_len, 1))
        
    
        checkins = np.zeros((batch_size, max_seq_len))
        for i, seq in enumerate(batch_seqs):
            seq_len = len(seq[1])
            to_pad = max_seq_len - seq_len
            if to_pad < 0:
                to_pad = 0
            checkins[i, to_pad:] = [x[0] for x in seq[1][:max_seq_len]]
       
        
        neg_checkins = np.zeros((batch_size, max_seq_len))
        for i in range(batch_size):
            for j in range(max_seq_len):
                while True:
                    random_venue = choice(allvenues)
                    if random_venue not in neg_checkins[i, :j]:
                        break
                neg_checkins[i,j] = random_venue
        
        #chop some of the data off, to leave a target
        X = [users, times[:,3+end:end], time_gaps[:,3+end:end], pos_distances[:,3+end:end], neg_distances[:,3+end:end], checkins[:,3+end:end], neg_checkins[:,3+end:end]]
        y = checkins[:,end]
        yield (X, y)        
         
def data_as_array(shuffle_every_epoch=True, status='train'):
    full_batch_size=len(sequences)
    gen = batch_generator(full_batch_size, shuffle_every_epoch, status)
    return next(gen)


our_X, our_y = data_as_array()
print([x.shape for x in X])
print(y.shape)
num_users = len(sequences)+1  #+1 for 1-indexed things
num_items = len(allvenues)+1
num_times = 1
latent_dim = 10
seq_length = max_seq_len-3
print('building model')
our_rec = Recommender(num_users, num_items, num_times, latent_dim, seq_length)



print('starting fit')
our_rec.model.fit(our_X, our_y)

#batch_size = 1
#n_epochs = 3
#rec.model.fit_generator(batch_generator(batch_size=batch_size), samples_per_epoch=len(sequences)/batch_size, nb_epoch=n_epochs, 
#                       verbose=0, callbacks=[TQDMNotebookCallback()])


rec.rank(users, checkins, times, time_gaps, pos_distances)

print(rec.model.summary())


