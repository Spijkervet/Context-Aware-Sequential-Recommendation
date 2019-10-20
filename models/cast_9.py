from modules import *

# Context Aware Sequential Transformer using a Sinusoidal Positional embedding
# Addition of the static positional encoding
# Concatenation of input context before the transformer
# Input context is also fed through a transformer before concatenation
class CAST9():
    def __init__(self, usernum, itemnum, ratingnum, args, reuse=None):

        if args.seed:
            tf.set_random_seed(args.seed)

        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))

        self.time_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))

        self.hours = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.days = tf.placeholder(tf.int32, shape=(None, args.maxlen))

        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)
        self.mask = mask

        # INPUT-CONTEXT AWARE
        print('INPUT-CONTEXT-AWARE MODULE')
        with tf.variable_scope("INPUT-CONTEXT", reuse=reuse):
            self.hours_seq, _ = embedding(self.hours,
                                          # anton's magic number (24 hours + zero padding)
                                          vocab_size=25,
                                          num_units=args.hidden_units,
                                          zero_pad=True,
                                          scale=True,
                                          l2_reg=args.l2_emb,
                                          scope="hours_embeddings",
                                          with_t=True,
                                          reuse=reuse)

            self.days_seq, _ = embedding(self.days,
                                         # anton's magic number (7 days + zero padding)
                                         vocab_size=8,
                                         num_units=args.hidden_units,
                                         zero_pad=True,
                                         scale=True,
                                         l2_reg=args.l2_emb,
                                         scope="days_embeddings",
                                         with_t=True,
                                         reuse=reuse)

            # Self-attention blocks
            # Build blocks
            for i in range(args.num_context_blocks):
                with tf.variable_scope("hours_seq_num_blocks_%d" % i):
                    # Self-attention
                    self.hours_seq, self.hours_weights = multihead_attention(self,
                                                                             queries=normalize(self.hours_seq),
                                                                             keys=self.hours_seq,
                                                                             num_units=args.hidden_units,
                                                                             num_heads=args.num_heads,
                                                                             dropout_rate=args.dropout_rate,
                                                                             is_training=self.is_training,
                                                                             causality=True,
                                                                             scope="self_attention")

                    # Feed forward
                    self.hours_seq = feedforward(normalize(self.hours_seq), num_units=[args.hidden_units, args.hidden_units],
                                            dropout_rate=args.dropout_rate, is_training=self.is_training)
                    self.hours_seq *= mask

            self.hours_seq = normalize(self.hours_seq)

            # Self-attention blocks
            # Build blocks
            for i in range(args.num_context_blocks):
                with tf.variable_scope("days_seq_num_blocks_%d" % i):
                    # Self-attention
                    self.days_seq, self.days_weights = multihead_attention(self,
                                                                           queries=normalize(self.days_seq),
                                                                           keys=self.days_seq,
                                                                           num_units=args.hidden_units,
                                                                           num_heads=args.num_heads,
                                                                           dropout_rate=args.dropout_rate,
                                                                           is_training=self.is_training,
                                                                           causality=True,
                                                                           scope="self_attention")

                    # Feed forward
                    self.days_seq = feedforward(normalize(self.days_seq), num_units=[args.hidden_units, args.hidden_units],
                                                 dropout_rate=args.dropout_rate, is_training=self.is_training)
                    self.days_seq *= mask
            self.days_seq = normalize(self.days_seq)


        # TEMPORAL CONTEXT-AWARE
        print('TEMPORAL CONTEXT-AWARE MODULE')
        with tf.variable_scope("TEMPORAL-CONTEXT", reuse=reuse):
            self.tseq, _ = embedding(self.time_seq,
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
            for i in range(args.num_context_blocks):
                with tf.variable_scope("timeseq_num_blocks_%d" % i):
                    # Self-attention
                    self.tseq, self.attention_weights = multihead_attention(self, queries=normalize(self.tseq),
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
                    self.tseq *= mask
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
            positional_embedding, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            self.seq += positional_embedding
            
            # CONCAT CONTEXTS
            self.concat_seq = tf.concat([self.seq, self.tseq, self.hours_seq, self.days_seq], axis=2)
            
            # Dropout
            self.concat_seq = tf.layers.dropout(self.concat_seq,
                                                rate=args.dropout_rate,
                                                training=tf.convert_to_tensor(self.is_training))

            # Go from concat -> 100x original embedding dimension
            self.seq = mlp(self.concat_seq, [self.concat_seq.get_shape()[2], args.hidden_units])

            # zero-pads back to 0
            self.seq *= self.mask

            # Self-attention blocks
            # Build blocks
            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):

                    # Self-attention
                    self.queries = normalize(self.seq)
                    self.keys = self.seq
                    self.seq, self.attention_weights = multihead_attention(self,
                                                                           queries=self.queries,
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

    def predict(self, sess, u, seq, item_idx, timeseq=None, hours_seq=None, days_seq=None):
        return sess.run([self.test_logits, self.attention_weights],
                        {self.u: u, self.input_seq: seq, self.time_seq: timeseq, self.test_item: item_idx, self.is_training: False})