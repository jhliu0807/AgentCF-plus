from collections import UserDict
from itertools import count
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    - Q (query), K (key) and V (value) are split into multiple heads (num_heads)
    - each tuple (q, k, v) are fed to scaled_dot_product_attention
    - all attention outputs are concatenated

    Args:
            attention_dim (int): Dimension of the attention embeddings.
            num_heads (int): Number of heads in the multi-head self-attention module.
            dropout_rate (float): Dropout probability.
    """

    def __init__(self, attention_dim, num_heads, dropout_rate):

        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        assert attention_dim % self.num_heads == 0
        self.dropout_rate = dropout_rate

        self.depth = attention_dim // self.num_heads

        self.Q = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.K = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.V = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, queries, keys):
        """Model forward pass.

        Args:
            queries (tf.Tensor): Tensor of queries.
            keys (tf.Tensor): Tensor of keys

        Returns:
            tf.Tensor: Output tensor
        """

        # Linear projections
        Q = self.Q(queries)  # (N, T_q, C)
        K = self.K(keys)  # (N, T_k, C)
        V = self.V(keys)  # (N, T_k, C)

        # --- MULTI HEAD ---
        # Split and concat, Q_, K_ and V_ are all (h*N, T_q, C/h)
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)

        # --- SCALED DOT PRODUCT ---
        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [self.num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(
            tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]
        )  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-(2 ** 32) + 1)
        # outputs, (h*N, T_q, T_k)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        # Future blinding (Causality)
        diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(
            diag_vals
        ).to_dense()  # (T_q, T_k)
        masks = tf.tile(
            tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]
        )  # (h*N, T_q, T_k)

        paddings = tf.ones_like(masks) * (-(2 ** 32) + 1)
        # outputs, (h*N, T_q, T_k)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking, query_masks (N, T_q)
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
        query_masks = tf.tile(query_masks, [self.num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(
            tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]
        )  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = self.dropout(outputs)

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # --- MULTI HEAD ---
        # concat heads
        outputs = tf.concat(
            tf.split(outputs, self.num_heads, axis=0), axis=2
        )  # (N, T_q, C)

        # Residual connection
        outputs += queries

        return outputs


class PointWiseFeedForward(tf.keras.layers.Layer):
    """
    Convolution layers with residual connection

    Args:
            conv_dims (list): List of the dimensions of the Feedforward layer.
            dropout_rate (float): Dropout probability.
    """

    def __init__(self, conv_dims, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv_dims = conv_dims
        self.dropout_rate = dropout_rate
        self.conv_layer1 = tf.keras.layers.Conv1D(
            filters=self.conv_dims[0], kernel_size=1, activation="relu", use_bias=True
        )
        self.conv_layer2 = tf.keras.layers.Conv1D(
            filters=self.conv_dims[1], kernel_size=1, activation=None, use_bias=True
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, x):
        """Model forward pass.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor
        """

        output = self.conv_layer1(x)
        output = self.dropout_layer(output)

        output = self.conv_layer2(output)
        output = self.dropout_layer(output)

        # Residual connection
        output += x

        return output


class EncoderLayer(tf.keras.layers.Layer):
    """
    Transformer based encoder layer

    Args:
            seq_max_len (int): Maximum sequence length.
            embedding_dim (int): Embedding dimension.
            attention_dim (int): Dimension of the attention embeddings.
            num_heads (int): Number of heads in the multi-head self-attention module.
            conv_dims (list): List of the dimensions of the Feedforward layer.
            dropout_rate (float): Dropout probability.
    """

    def __init__(
        self,
        seq_max_len,
        embedding_dim,
        attention_dim,
        num_heads,
        conv_dims,
        dropout_rate,
    ):
        """Initialize parameters.

        
        """
        super(EncoderLayer, self).__init__()

        self.seq_max_len = seq_max_len
        self.embedding_dim = embedding_dim

        self.mha = MultiHeadAttention(attention_dim, num_heads, dropout_rate)
        self.ffn = PointWiseFeedForward(conv_dims, dropout_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.layer_normalization = LayerNormalization(
            self.seq_max_len, self.embedding_dim, 1e-08
        )

    def call_(self, x, training, mask):
        """Model forward pass.

        Args:
            x (tf.Tensor): Input tensor.
            training (tf.Tensor): Training tensor.
            mask (tf.Tensor): Mask tensor.

        Returns:
            tf.Tensor: Output tensor
        """

        attn_output = self.mha(queries=self.layer_normalization(x), keys=x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # feed forward network
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # (batch_size, input_seq_len, d_model)

        # masking
        out2 *= mask

        return out2

    def call(self, x, training, mask):
        """Model forward pass.

        Args:
            x (tf.Tensor): Input tensor.
            training (tf.Tensor): Training tensor.
            mask (tf.Tensor): Mask tensor.

        Returns:
            tf.Tensor: Output tensor
        """

        x_norm = self.layer_normalization(x)
        attn_output = self.mha(queries=x_norm, keys=x)
        attn_output = self.ffn(attn_output)
        out = attn_output * mask

        return out


class Encoder(tf.keras.layers.Layer):
    """
    Invokes Transformer based encoder with user defined number of layers

    Args:
            num_layers (int): Number of layers.
            seq_max_len (int): Maximum sequence length.
            embedding_dim (int): Embedding dimension.
            attention_dim (int): Dimension of the attention embeddings.
            num_heads (int): Number of heads in the multi-head self-attention module.
            conv_dims (list): List of the dimensions of the Feedforward layer.
            dropout_rate (float): Dropout probability.
    """

    def __init__(
        self,
        num_layers,
        seq_max_len,
        embedding_dim,
        attention_dim,
        num_heads,
        conv_dims,
        dropout_rate,
    ):
        """Initialize parameters.

        
        """
        super(Encoder, self).__init__()

        self.num_layers = num_layers

        self.enc_layers = [
            EncoderLayer(
                seq_max_len,
                embedding_dim,
                attention_dim,
                num_heads,
                conv_dims,
                dropout_rate,
            )
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        """Model forward pass.

        Args:
            x (tf.Tensor): Input tensor.
            training (tf.Tensor): Training tensor.
            mask (tf.Tensor): Mask tensor.

        Returns:
            tf.Tensor: Output tensor
        """

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)

        return x  # (batch_size, input_seq_len, d_model)


class LayerNormalization(tf.keras.layers.Layer):
    """
    Layer normalization using mean and variance
    gamma and beta are the learnable parameters

    Args:
            seq_max_len (int): Maximum sequence length.
            embedding_dim (int): Embedding dimension.
            epsilon (float): Epsilon value.
    """

    def __init__(self, seq_max_len, embedding_dim, epsilon):
        """Initialize parameters.

        Args:
            seq_max_len (int): Maximum sequence length.
            embedding_dim (int): Embedding dimension.
            epsilon (float): Epsilon value.
        """
        super(LayerNormalization, self).__init__()
        self.seq_max_len = seq_max_len
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        self.params_shape = (self.seq_max_len, self.embedding_dim)
        g_init = tf.ones_initializer()
        self.gamma = tf.Variable(
            initial_value=g_init(shape=self.params_shape, dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.beta = tf.Variable(
            initial_value=b_init(shape=self.params_shape, dtype="float32"),
            trainable=True,
        )

    def call(self, x):
        """Model forward pass.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor
        """
        mean, variance = tf.nn.moments(x, [-1], keepdims=True)
        normalized = (x - mean) / ((variance + self.epsilon) ** 0.5)
        output = self.gamma * normalized + self.beta
        return output


class SASREC(tf.keras.Model):
    """Self-Attentive Sequential Recommendation Using Transformer    

    Keyword Args:
        item_num (int): Number of items in the dataset.
        seq_max_len (int): Maximum number of items in user history.
        num_blocks (int): Number of Transformer blocks to be used.
        embedding_dim (int): Item embedding dimension.
        attention_dim (int): Transformer attention dimension.
        attention_num_heads (int): Transformer attention head.
        conv_dims (list): List of the dimensions of the Feedforward layer.
        dropout_rate (float): Dropout rate.
        l2_reg (float): Coefficient of the L2 regularization.

    Attributes:
        epoch (int): Epoch of trained model.
        best_score (float): Best validation HR@10 score while training.
        val_users (list): User list for validation.
        history (pd.DataFrame): Train history containing epoch, NDCG@10, and HR@10.
    """
    
    def __init__(self, **kwargs):

        super(SASREC, self).__init__()

        self.epoch = 0
        self.best_score=0
        self.val_users = []
        self.history = pd.DataFrame(columns=['epoch','NDCG@10','HR@10'])

        self.item_num = kwargs.get("item_num", None)
        self.seq_max_len = kwargs.get("seq_max_len", 100)
        self.num_blocks = kwargs.get("num_blocks", 2)
        self.embedding_dim = kwargs.get("embedding_dim", 100)
        self.attention_dim = kwargs.get("attention_dim", 100)
        self.attention_num_heads = kwargs.get("attention_num_heads", 1)
        self.conv_dims = kwargs.get("conv_dims", [100, 100])
        self.dropout_rate = kwargs.get("dropout_rate", 0.5)
        self.l2_reg = kwargs.get("l2_reg", 0.0)

        self.item_embedding_layer = tf.keras.layers.Embedding(
            self.item_num + 1,
            self.embedding_dim,
            name="item_embeddings",
            mask_zero=True,
            embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg),
        )

        self.positional_embedding_layer = tf.keras.layers.Embedding(
            self.seq_max_len,
            self.embedding_dim,
            name="positional_embeddings",
            mask_zero=False,
            embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg),
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        self.encoder = Encoder(
            self.num_blocks,
            self.seq_max_len,
            self.embedding_dim,
            self.attention_dim,
            self.attention_num_heads,
            self.conv_dims,
            self.dropout_rate,
        )
        self.mask_layer = tf.keras.layers.Masking(mask_value=0)
        self.layer_normalization = LayerNormalization(
            self.seq_max_len, self.embedding_dim, 1e-08
        )

    def embedding(self, input_seq):
        """Compute the sequence and positional embeddings.

        Args:
            input_seq (tf.Tensor): Input sequence

        Returns:
            tf.Tensor: Sequence embeddings
            tf.Tensor: Positional embeddings
        """

        seq_embeddings = self.item_embedding_layer(input_seq)
        seq_embeddings = seq_embeddings * (self.embedding_dim ** 0.5)

        # FIXME
        positional_seq = tf.expand_dims(tf.range(tf.shape(input_seq)[1]), 0)
        positional_seq = tf.tile(positional_seq, [tf.shape(input_seq)[0], 1])
        positional_embeddings = self.positional_embedding_layer(positional_seq)

        return seq_embeddings, positional_embeddings

    def call(self, x, training):
        """Model forward pass.

        Args:
            x (tf.Tensor): Input tensor.
            training (tf.Tensor): Training tensor.

        Returns:
            tf.Tensor: Logits of the positive examples
            tf.Tensor: Logits of the negative examples
            tf.Tensor: Mask for nonzero targets
        """

        input_seq = x["input_seq"]
        pos = x["positive"]
        neg = x["negative"]

        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        seq_embeddings, positional_embeddings = self.embedding(input_seq)

        # add positional embeddings
        seq_embeddings += positional_embeddings

        # dropout
        seq_embeddings = self.dropout_layer(seq_embeddings)

        # masking
        seq_embeddings *= mask

        # --- ATTENTION BLOCKS ---
        seq_attention = seq_embeddings
        seq_attention = self.encoder(seq_attention, training, mask)
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, d)

        # --- PREDICTION LAYER ---
        # user's sequence embedding
        pos = self.mask_layer(pos)
        neg = self.mask_layer(neg)

        pos = tf.reshape(pos, [tf.shape(input_seq)[0] * self.seq_max_len])
        neg = tf.reshape(neg, [tf.shape(input_seq)[0] * self.seq_max_len])
        pos_emb = self.item_embedding_layer(pos)
        neg_emb = self.item_embedding_layer(neg)
        seq_emb = tf.reshape(
            seq_attention,
            [tf.shape(input_seq)[0] * self.seq_max_len, self.embedding_dim],
        )  # (b*s, d)

        pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        pos_logits = tf.expand_dims(pos_logits, axis=-1)  # (bs, 1)
        # pos_prob = tf.keras.layers.Dense(1, activation='sigmoid')(pos_logits)  # (bs, 1)

        neg_logits = tf.expand_dims(neg_logits, axis=-1)  # (bs, 1)
        # neg_prob = tf.keras.layers.Dense(1, activation='sigmoid')(neg_logits)  # (bs, 1)

        # output = tf.concat([pos_logits, neg_logits], axis=0)

        # masking for loss calculation
        istarget = tf.reshape(
            tf.cast(tf.not_equal(pos, 0), dtype=tf.float32),
            [tf.shape(input_seq)[0] * self.seq_max_len],
        )

        return pos_logits, neg_logits, istarget

    def predict(self, inputs,neg_cand_n):
        """Returns the logits for the test items.

        Args:
            inputs (tf.Tensor): Input tensor.
            neg_cand_n: num of negative candidates
        Returns:
            tf.Tensor:Output tensor
        """
        training = False
        input_seq = inputs["input_seq"]
        candidate = inputs["candidate"]

        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        seq_embeddings, positional_embeddings = self.embedding(input_seq)
        seq_embeddings += positional_embeddings
        # seq_embeddings = self.dropout_layer(seq_embeddings)
        seq_embeddings *= mask
        seq_attention = seq_embeddings
        seq_attention = self.encoder(seq_attention, training=training, mask=mask)
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, d)
        seq_emb = tf.reshape(
            seq_attention,
            [tf.shape(input_seq)[0] * self.seq_max_len, self.embedding_dim],
        )  # (b*s, d)
        # print(candidate)
        candidate_emb = self.item_embedding_layer(candidate)  # (b, s, d)
        candidate_emb = tf.transpose(candidate_emb, perm=[0, 2, 1])  # (b, d, s)

        test_logits = tf.matmul(seq_emb, candidate_emb)
        # (200, 100) * (1, 101, 100)'

        test_logits = tf.reshape(
            test_logits,
            [tf.shape(input_seq)[0], self.seq_max_len, 1+neg_cand_n],
        )  # (1, 50, 1+neg_can)
        test_logits = test_logits[:, -1, :]  # (1, 101)
        return test_logits

    def loss_function(self, pos_logits, neg_logits, istarget):
        """Losses are calculated separately for the positive and negative
        items based on the corresponding logits. A mask is included to
        take care of the zero items (added for padding).

        Args:
            pos_logits (tf.Tensor): Logits of the positive examples.
            neg_logits (tf.Tensor): Logits of the negative examples.
            istarget (tf.Tensor): Mask for nonzero targets.

        Returns:
            float: loss
        """

        pos_logits = pos_logits[:, 0]
        neg_logits = neg_logits[:, 0]

        # ignore padding items (0)
        # istarget = tf.reshape(
        #     tf.cast(tf.not_equal(self.pos, 0), dtype=tf.float32),
        #     [tf.shape(self.input_seq)[0] * self.seq_max_len],
        # )
        # for logits
        loss = tf.reduce_sum(
            -tf.math.log(tf.math.sigmoid(pos_logits) + 1e-24) * istarget
            - tf.math.log(1 - tf.math.sigmoid(neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)

        # for probabilities
        # loss = tf.reduce_sum(
        #         - tf.math.log(pos_logits + 1e-24) * istarget -
        #         tf.math.log(1 - neg_logits + 1e-24) * istarget
        # ) / tf.reduce_sum(istarget)
        reg_loss = tf.compat.v1.losses.get_regularization_loss()
        # reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        # loss += sum(reg_losses)
        loss += reg_loss

        return loss

    def create_combined_dataset(self, u, seq, pos, neg):
        """
        function to create model inputs from sampled batch data.
        This function is used during training.
        """
        inputs = {}
        seq = tf.keras.preprocessing.sequence.pad_sequences(
            seq, padding="pre", truncating="pre", maxlen=self.seq_max_len
        )
        pos = tf.keras.preprocessing.sequence.pad_sequences(
            pos, padding="pre", truncating="pre", maxlen=self.seq_max_len
        )
        neg = tf.keras.preprocessing.sequence.pad_sequences(
            neg, padding="pre", truncating="pre", maxlen=self.seq_max_len
        )

        inputs["users"] = np.expand_dims(np.array(u), axis=-1)
        inputs["input_seq"] = seq
        inputs["positive"] = pos
        inputs["negative"] = neg

        target = np.concatenate(
            [
                np.repeat(1, seq.shape[0] * seq.shape[1]),
                np.repeat(0, seq.shape[0] * seq.shape[1]),
            ],
            axis=0,
        )
        target = np.expand_dims(target, axis=-1)
        return inputs, target

    def train(self, dataset, sampler, num_epochs=10,batch_size=128,lr=0.001,\
            val_epoch=5,val_target_user_n=1000,target_item_n=-1,auto_save=False,\
            path='./',exp_name='SASRec_exp'):
        
        """High level function for model training as well as evaluation on the validation and test dataset

        Args:
            dataset (:obj:`util.SASRecDataSet`): SASRecDataSet containing users-item interaction history.
            sampler (:obj:`util.WarpSampler`): WarpSampler.
            num_epochs (int, optional): Epoch. Defaults to 10.
            batch_size (int, optional): Batch size. Defaults to 128.
            lr (float, optional): Learning rate. Defaults to 0.001.
            val_epoch (int, optional): Validation term. Defaults to 5.
            val_target_user_n (int, optional): Number of randomly sampled users to conduct validation. Defaults to 1000.
            target_item_n (int, optional): Size of candidate. Defaults to -1, which means all.
            auto_save (bool, optional): If true, save model with best validation score. Defaults to False.
            path (str, optional): Path to save model.
            exp_name (str, optional): Experiment name.

        """
        
        num_steps = int(len(dataset.user_train) / batch_size)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7
        )

        loss_function = self.loss_function

        train_loss = tf.keras.metrics.Mean(name="train_loss")

        train_step_signature = [
            {
                "users": tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
                "input_seq": tf.TensorSpec(
                    shape=(None, self.seq_max_len), dtype=tf.int64
                ),
                "positive": tf.TensorSpec(
                    shape=(None, self.seq_max_len), dtype=tf.int64
                ),
                "negative": tf.TensorSpec(
                    shape=(None, self.seq_max_len), dtype=tf.int64
                ),
            },
            tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
        ]

        @tf.function(input_signature=train_step_signature)
        def train_step(inp, tar):
            with tf.GradientTape() as tape:
                pos_logits, neg_logits, loss_mask = self(inp, training=True)
                loss = loss_function(pos_logits, neg_logits, loss_mask)

            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            train_loss(loss)
            return loss
        
        

        for epoch in range(1, num_epochs + 1):

            print(f'epoch {epoch} / {num_epochs} -----------------------------')
            
            self.epoch = epoch
            step_loss = []
            train_loss.reset_state()
            for step in tqdm(
                range(num_steps), total=num_steps, ncols=70, leave=False, unit="b",
                # disable= ~progress_bar
            ):

                u, seq, pos, neg = sampler.next_batch()

                inputs, target = self.create_combined_dataset(u, seq, pos, neg)

                loss = train_step(inputs, target)
                step_loss.append(loss)

            if epoch % val_epoch == 0:                
                print("Evaluating...")
                t_test = self.evaluate(dataset,target_user_n=val_target_user_n,target_item_n=target_item_n,is_val=True)
                print(
                    f"epoch: {epoch}, test (NDCG@10: {t_test[0]}, HR@10: {t_test[1]})"
                )
                self.history.loc[len(self.history)] = [epoch,t_test[0],t_test[1]]
                
                if t_test[1] > self.best_score:
                    self.best_score = t_test[1]
                    if auto_save:
                        self.save(path,exp_name)
                        print('best score model updated and saved')
        
        if auto_save:
            self.history.to_csv(f'{path}/{exp_name}/{exp_name}_train_log.csv',index=False)
        
        return

    def evaluate(self, dataset,target_user_n=1000,target_item_n=-1,rank_threshold=10,is_val=False):
        """Evaluate model on validation set or test set

        Args:
            dataset (:obj:`SASRecDataSet`): SASRecDataSet containing users-item interaction history.
            target_user_n (int, optional): Number of randomly sampled users to evaluate. Defaults to 1000.
            target_item_n (int, optional): Size of candidate. Defaults to -1, which means all.
            rank_threshold (int, optional): k value in NDCG@k and HR@k. Defaults to 10.
            is_val (bool, optional): If true, evaluate on validation set. If False, evaluate on test set. Defaults to False.

        Returns:
            float: NDCG@k
            float: HR@k
        """

        usernum = dataset.usernum
        itemnum = dataset.itemnum
        all = dataset.User
        train = dataset.user_train  # removing deepcopy
        valid = dataset.user_valid
        test = dataset.user_test

        NDCG = 0.0
        HT = 0.0
        valid_user = 0.0
        
        if len(self.val_users) == 0:
            self.sample_val_users(dataset,target_user_n)

        for u in tqdm(self.val_users, ncols=70, leave=False, unit="b"):

            if len(train[u]) < 1 or len(test[u]) < 1:
                continue

            seq = np.zeros([self.seq_max_len], dtype=np.int32)
            idx = self.seq_max_len - 1

            if is_val:                
                item_idx = [valid[u][0]]
            else:
                seq[idx] = valid[u][0]
                idx -= 1
                item_idx = [test[u][0]]

            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break

            rated = set(all[u])

            if (target_item_n == -1):
              item_idx=item_idx+list(set(range(1,itemnum+1)).difference(rated))
            
            elif type(target_item_n)==int:
              for _ in range(target_item_n):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)
            
            elif type(target_item_n)==float:
              for _ in range(round(itemnum*target_item_n)):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

            else:
              raise            
            
            inputs = {}
            inputs["user"] = np.expand_dims(np.array([u]), axis=-1)
            inputs["input_seq"] = np.array([seq])
            inputs["candidate"] = np.array([item_idx])
            # print(inputs)

            # inverse to get descending sort
            predictions = -1.0 * self.predict(inputs, len(item_idx)-1)
            predictions = np.array(predictions)
            predictions = predictions[0]
            # print('predictions:', predictions)

            rank = predictions.argsort().argsort()[0]
            # print('rank:', rank)

            valid_user += 1

            if rank < rank_threshold:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1

        return NDCG / valid_user, HT / valid_user

    def recommend_item(self, dataset, user_map_dict,user_id_list, target_item_n=-1,top_n=10,exclude_purchased=True,is_test=False):
        """Recommend items to user

        Args:
            dataset (:obj:`util.SASRecDataSet`): SASRecDataSet containing users-item interaction history.
            user_map_dict (dict): Dict { user_id : encoded_user_label , ... }
            user_id_list (list): User list to predict.
            target_item_n (int, optional): Size of candidate. Defaults to -1, which means all.
            top_n (int, optional): Number of items to recommend. Defaults to 10.
            exclude_purchased (bool, optional): If true, exclude already purchased item from candidate. Defaults to True.
            is_test (bool, optional): If true, exclude the last item from each user's sequence. Defaults to False.

        Returns:
            pd.DataFrame: recommended items for users
        """
        all = dataset.User
        itemnum = dataset.itemnum
        users = [user_map_dict[u] for u in user_id_list]
        inv_user_map = {v: k for k, v in user_map_dict.items()}
        return_dict={}

        for u in tqdm(users):
            seq = np.zeros([self.seq_max_len], dtype=np.int32)
            idx = self.seq_max_len - 1

            list_to_seq = all[u] if not is_test else all[u][:-1]
            for i in reversed(list_to_seq):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break

            if exclude_purchased: 
                rated = set(all[u]) 
            else: 
                rated = set()
            
            # make empty candidate list
            item_idx=[]

            if (target_item_n == -1):
                item_idx=item_idx+list(set(range(1,itemnum+1)).difference(rated))
            
            elif type(target_item_n)==int:
                for _ in range(target_item_n):
                    t = np.random.randint(1, itemnum + 1)
                    while t in rated:
                        t = np.random.randint(1, itemnum + 1)
                    item_idx.append(t)
    
            elif type(target_item_n)==float:
                for _ in range(round(itemnum*target_item_n)):
                    t = np.random.randint(1, itemnum + 1)
                    while t in rated:
                        t = np.random.randint(1, itemnum + 1)
                    item_idx.append(t)

            else:
                raise
        
            inputs = {}
            inputs["user"] = np.expand_dims(np.array([u]), axis=-1)
            inputs["input_seq"] = np.array([seq])
            inputs["candidate"] = np.array([item_idx])

            predictions = self.predict(inputs, len(item_idx)-1)
            predictions = np.array(predictions)
            predictions = predictions[0]

            pred_dict = {v : predictions[i] for i,v in enumerate(item_idx)}
            pred_dict = sorted(pred_dict.items(), key = lambda item: item[1], reverse = True)
            top_list = pred_dict[:top_n]

            return_dict[inv_user_map[u]] = top_list

        return return_dict
    
    def old_get_user_item_score(self, dataset, user_map_dict,item_map_dict,user_id_list, item_list,is_test=False):
        """
        Deprecated
        """
        all = dataset.User
        users = [user_map_dict[u] for u in user_id_list]
        items = [item_map_dict[i] for i in item_list]
        # inv_user_map = {v: k for k, v in user_map_dict.items()}
        # inv_item_map = {v: k for k, v in item_map_dict.items()}        
        score_dict = {i:[] for i in item_list}
        
        for u in tqdm(users,unit=' User',desc='Getting Scores for each user ...'):
                
            seq = np.zeros([self.seq_max_len], dtype=np.int32)
            idx = self.seq_max_len - 1

            list_to_seq = all[u] if not is_test else all[u][:-1]
            for i in reversed(list_to_seq):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break
        
            inputs = {}
            inputs["user"] = np.expand_dims(np.array([u]), axis=-1)
            inputs["input_seq"] = np.array([seq])
            inputs["candidate"] = np.array([items])

            predictions = self.predict(inputs, len(items)-1)
            predictions = np.array(predictions)
            predictions = predictions[0]

            # pred_dict = {inv_item_map[v] : predictions[i] for i,v in enumerate(items)}

            for i,v in enumerate(item_list):
                score_dict[v].append(predictions[i])                      

        return_df = pd.DataFrame({
            'user_id':user_id_list,
        })
        
        for k in score_dict:
            return_df[k] = score_dict[k]
        
        return_df = return_df.sort_values(by='user_id').reset_index(drop=True)

        return return_df
        
    def get_user_item_score(self,dataset,user_id_list, item_list,user_map_dict,item_map_dict,batch_size=128):
        """Get item score for each user on batch

        Args:
            dataset (:obj:`SASRecDataSet`): SASRecDataSet containing users-item interaction history.
            user_id_list (list): User list to predict.
            item_list (list): Item list to predict
            user_map_dict (dict): Dict { user_id : encoded_user_label , ... }
            item_map_dict (dict): Dict { item : encoded_item_label , ... }
            batch_size (int, optional): Batch size. Defaults to 128.

        Raises:
            Exception: Batch_size must be smaller than user id list size.

        Returns:
            pd.DataFrame: user-item score
        """

        if batch_size > len(user_id_list):
            raise Exception('batch_size must be smaller than user_id_list size')

        user_history = dataset.User
        inv_user_map = {v: k for k, v in user_map_dict.items()}
        cand = [item_map_dict[i] for i in item_list]

        if len(user_id_list)%batch_size ==0:
            num_steps = int(len(user_id_list)/batch_size)
        else:
            num_steps = int(len(user_id_list)/batch_size) + 1
        # num_steps = len(user_id_list)/batch_size
        # num_steps = num_steps if num_steps%1 == 0 else int(num_steps)+1        
        print(num_steps)
        score_dict = dict()
        
        start_index=0

        for step in tqdm(
                range(num_steps), leave=True, unit="batch",
            ):

            # get user batch
            u = user_id_list[start_index:start_index+batch_size] if step != num_steps-1 \
                else user_id_list[start_index:]

            start_index+=batch_size     
            
            u = [user_map_dict[user] for user in u] 

            # get sequence batch
            seq = [user_history[user] for user in u]

            inputs = self.create_combined_dataset_pred(tuple(u),tuple(seq),cand)
            # print(inputs)

            predictions = self.batch_predict(inputs, len(cand)-1)
            predictions = np.array(predictions)

            for i in range(len(u)):
                score_dict[inv_user_map[u[i]]]=predictions[i] 


        return_df = pd.DataFrame(list(score_dict.items()),columns = ['user_id','score_array']).set_index('user_id',drop=True)
        # print(return_df)
        return_df[item_list] = pd.DataFrame(return_df['score_array'].tolist(), index= return_df.index)
        return_df = return_df.drop('score_array',axis=1).reset_index().sort_values(by='user_id').reset_index(drop=True)

        return return_df
    

    def create_combined_dataset_pred(self,u,seq,cand):
        """
        function to create model inputs from sampled batch data.
        This function is used during predicting on batch.
        """
        inputs = {}
        seq = tf.keras.preprocessing.sequence.pad_sequences(
            seq, padding="pre", truncating="pre", maxlen=self.seq_max_len
        )

        inputs["users"] = np.expand_dims(np.array(u), axis=-1)
        inputs["input_seq"] = seq
        inputs['candidate'] = np.array([cand])

        return inputs
    

    def batch_predict(self, inputs,cand_n):
        """Returns the logits for the item candidates.

        Args:
            inputs (tf.Tensor): Input tensor.
            cand_n (int): Num of candidates.

        Returns:
            tf.Tensor: Output tensor
        """
        training = False
        input_seq = inputs["input_seq"]
        candidate = inputs["candidate"]

        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        seq_embeddings, positional_embeddings = self.embedding(input_seq)
        seq_embeddings += positional_embeddings
        # seq_embeddings = self.dropout_layer(seq_embeddings)
        seq_embeddings *= mask
        seq_attention = seq_embeddings
        seq_attention = self.encoder(seq_attention, training=training, mask=mask)
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, d)
        seq_emb = tf.reshape(
            seq_attention,
            [tf.shape(input_seq)[0] * self.seq_max_len, self.embedding_dim],
        )  # (b*s, d)
        # print(candidate)
        candidate_emb = self.item_embedding_layer(candidate)  # (b, s, d)
        candidate_emb = tf.transpose(candidate_emb, perm=[0, 2, 1])  # (b, d, s)

        test_logits = tf.matmul(seq_emb, candidate_emb)
        # (200, 100) * (1, 101, 100)'

        test_logits = tf.reshape(
            test_logits,
            [tf.shape(input_seq)[0], self.seq_max_len, 1+cand_n],
        )  # (1, 50, 1+can)
        test_logits = test_logits[:, -1, :]  # (1, 101)
        return test_logits

    
    def save(self,path, exp_name='sas_experiment'):
        """Save trained SASRec Model

        Args:
            path (str): Path to save model.
            exp_name (str): Experiment name.
        
        Examples:
            >>> model.save(path, exp_name)       
        """
        
        # make dir
        if not os.path.exists(f'{path}/{exp_name}'):
            os.mkdir(f'{path}/{exp_name}')

        self.save_weights(f'{path}/{exp_name}/{exp_name}_weights.weights.h5') # save trained weights
        # arg_list = ['item_num','seq_max_len','num_blocks','embedding_dim','attention_dim','attention_num_heads','dropout_rate','conv_dims','l2_reg','history']
        arg_list = ['item_num','seq_max_len','num_blocks','embedding_dim','attention_dim','attention_num_heads','dropout_rate']
        dict_to_save = {a: self.__dict__[a] for a in arg_list}

        with open(f'{path}/{exp_name}/{exp_name}_model_args','wb') as f:
            pickle.dump(dict_to_save, f)
        
        if not os.path.isfile(f'{path}/{exp_name}/{exp_name}_save_log.txt'): 
            with open(f'{path}/{exp_name}/{exp_name}_save_log.txt','w') as f:
                f.writelines(f'Model args: {dict_to_save}\n')
                f.writelines(f'[epoch {self.epoch}] Best HR@10 score: {self.best_score}\n')
        else:
            with open(f'{path}/{exp_name}/{exp_name}_save_log.txt','a') as f:
                f.writelines(f'[epoch {self.epoch}] Best HR@10 score: {self.best_score}\n')
        
        return
    

    def sample_val_users(self,dataset,target_user_n):
        """Sample users for validation

        Args:
            dataset (:obj:`SASRecDataSet`): SASRec dataset used for training
            target_user_n (int): Number of users to sample
        """
        usernum = dataset.usernum
        if usernum > target_user_n:
            self.val_users = random.sample(range(1, usernum + 1), target_user_n)
        else:
            self.val_users = range(1, usernum + 1)
