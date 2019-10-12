from modules import *


class FutureCAST():
    def __init__(self, usernum, itemnum, args, reuse=None):

        if args.seed:
            tf.set_random_seed(args.seed)

        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))

        self.time_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        
        self.input_context = tf.placeholder(tf.int32, shape=(None, args.maxlen))

        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)
        self.mask = mask

        # Mask of time sequence data
        time_seq_mask = tf.expand_dims(tf.to_float(tf.not_equal(self.time_seq, 0)), -1)
        self.time_seq_mask = time_seq_mask

        # Mask of input sequence data
        input_context_seq_mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_context, 0)), -1)
        self.input_context_seq_mask = input_context_seq_mask 

        # CONTEXT-AWARE
        print('CONTEXT-AWARE MODULE')
        with tf.variable_scope("CONTEXT", reuse=reuse):
            # Time sequence encoding ('timestamps -> positional vector')
            # TODO: Either set encoding dims to args.max_time_interval or hidden_units, or use embedding()
            self.tseq_enc = timeseq_encoding(self.time_seq, args.max_bins+1)
            # self.tseq = timeseq_encoding(self.time_seq, args.hidden_units)
            self.tseq, item_emb_table = embedding(self.time_seq,
                                        vocab_size=args.max_bins+1,
                                        num_units=args.hidden_units,
                                        zero_pad=True,
                                        scale=True,
                                        l2_reg=args.l2_emb,
                                        scope="time_embeddings",
                                        with_t=True,
                                        reuse=reuse)

            # Self-attention blocks
            # Build blocks
            for i in range(args.num_blocks):
                with tf.variable_scope("timeseq_num_blocks_%d" % i):
                    # Self-attention
                    self.timeseq_queries = normalize(self.tseq)
                    self.timeseq_keys = self.tseq
                    self.tseq = multihead_attention(self, queries=normalize(self.tseq),
                                                    keys=self.tseq,
                                                    num_units=args.hidden_units,
                                                    num_heads=args.num_heads,
                                                    dropout_rate=args.dropout_rate,
                                                    is_training=self.is_training,
                                                    causality=True,
                                                    scope="self_attention")

                    # Feed forward
                    self.tseq = feedforward(normalize(self.tseq), num_units=[args.hidden_units, args.hidden_units],
                                            dropout_rate=args.dropout_rate, is_training=self.is_training)
                    self.tseq *= time_seq_mask
            self.tseq = normalize(self.tseq)

        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            self.item_emb_table = item_emb_table

            # Positional Encoding
            # positional_encoding(dim, sentence_length, dtype=tf.float32)
            t = positional_encoding(
                args.hidden_units,
                args.maxlen,
            )

            # Sinusoidal Positional Embedding
            self.seq += t

            # CONTEXT-AWARE MODULE
            self.seq += self.tseq

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask
            
            # Self-attention blocks
            # Build blocks
            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):

                    # Self-attention
                    self.queries = normalize(self.seq)
                    self.keys = self.seq
                    self.seq = multihead_attention(self, queries=self.queries,
                                                   keys=self.keys,
                                                   num_units=args.hidden_units,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                           dropout_rate=args.dropout_rate, is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq)

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])

        self.test_item = tf.placeholder(tf.int32, shape=(101))
        test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, 101])
        self.test_logits = self.test_logits[:, -1, :]

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        self.loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        tf.summary.scalar('TRAIN/loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('TRAIN/auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('TEST/test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq, timeseq, input_context_seq, item_idx):
        return sess.run(self.test_logits,
                        {self.u: u, self.input_seq: seq, self.time_seq: timeseq, self.input_context: input_context_seq, self.test_item: item_idx, self.is_training: False})

