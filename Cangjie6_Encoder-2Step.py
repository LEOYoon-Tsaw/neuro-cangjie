# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from PIL import ImageFont, ImageDraw, Image
from fontTools.ttLib import TTFont
from macrotoolchain import Data, Graph, plot 

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import time
import datetime
import json
import os


# -

# ## Load Data

class Glyph(object):
    # transform character to bitmap
    def __init__(self, fonts, size=64):
        # load fonts, size. We will use 2 fonts for all CJK characters, so keep 2 codepoint books.
        self.codepoints = [set()] * len(fonts)
        self.size = int(size * 0.8)
        self.size_img = size
        self.pad = (size - self.size) // 2
        self.fonts = [ImageFont.truetype(f, self.size) for f in fonts]
        # use a cache to reduce computation if duplicated characters encountered.
        self.cache = {}
        for cp, font in zip(self.codepoints, fonts):
            font = TTFont(font)
            # store codepoints in font cmap into self.codepoints
            for cmap in font['cmap'].tables:
                if not cmap.isUnicode():
                    continue
                for k in cmap.cmap:
                    cp.add(k)
    
    def draw(self, ch):
        if ch in self.cache:
            return self.cache[ch]
        # search among fonts, use the first found
        exist = False
        for i in range(len(self.codepoints)):
            if ord(ch) in self.codepoints[i]:
                font = self.fonts[i]
                exist = True
                break
        if not exist:
            return None

        img = Image.new('L', (self.size_img, self.size_img), 0)
        draw = ImageDraw.Draw(img)
        (width, baseline), (offset_x, offset_y) = font.font.getsize(ch)
        draw.text((self.pad - offset_x, self.pad - offset_y + 4), ch, font=font, fill=255, stroke_fill=255) 
        img_array = np.array(img.getdata(), dtype='float32').reshape((self.size_img, self.size_img)) / 255
        self.cache[ch] = img_array

        return img_array


glyphbook = Glyph(['data/fonts/HanaMinA.otf', 'data/fonts/HanaMinB.otf'])

code_chart = pd.read_csv('data/cangjie6.txt', delimiter='\t', header=None, names=['Char', 'Code'], 
                        keep_default_na=False)


def preprocess_chart(chart):
    glyphs = []
    codes = []
    for char, code in chart.values:
        glyph = glyphbook.draw(char)
        if glyph is not None:
            glyphs.append(glyph)
            codes.append(code)
    return np.expand_dims(np.array(glyphs), -1), np.array(codes)


VOCAB = 28
def tokenizer(code_table):
    # Cangjie code consists only of a-z, with maximum length of 5, minimum of 1
    # start with 0, a-z are 1-26, end and padding are 27
    tokens = np.expand_dims(np.zeros(code_table.shape, dtype='int64'), -1)
    code_index = list(map(lambda x: list(map(lambda y: ord(y) - 96, list(x))) + [27] * (5-len(x)), code_table))
    tokens = np.append(tokens, np.array(code_index), axis=-1)
    return tokens


glyphs, codes = preprocess_chart(code_chart)
tokens = tokenizer(codes)
lengths = np.array([len(list(filter(lambda i: i < 27 and i > 0, x))) for x in tokens])
lengths = np.array([np.identity(5)[i-1] for i in lengths], dtype='int64')
del code_chart, codes

train_glyphs, validation_glyphs, train_tokens, validation_tokens, train_lengths, validation_lengths = \
train_test_split(glyphs, tokens, lengths, test_size=0.1)
del glyphs, tokens, lengths

BATCH_SIZE = 128
dataset = tf.data.Dataset.from_tensor_slices((train_glyphs, train_tokens, train_lengths))
dataset = dataset.shuffle(train_glyphs.shape[0]).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# ## Model

class Res_CNN(tf.keras.Model):
    def __init__(self, feature_dim, kernel_size):
        super(Res_CNN, self).__init__()
        self.cnn1 = tf.keras.layers.Convolution2D(feature_dim, kernel_size)
        self.cnn2 = tf.keras.layers.Convolution2D(feature_dim, kernel_size, padding='same')
        self.cnn3 = tf.keras.layers.Convolution2D(feature_dim, kernel_size, padding='same')
        
    def call(self, x):
        x = self.cnn1(x)
        x_identity = tf.identity(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = tf.nn.relu(x + x_identity)
        return x


class CNN_Encoder(tf.keras.Model):
    # This is essentially a CNN layer, 
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.res_cnn1 = Res_CNN(embedding_dim // 16, (3, 3))
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D((2, 2))
        self.res_cnn2 = Res_CNN(embedding_dim // 4, (3, 3))
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPool2D((2, 2))
        self.res_cnn3 = Res_CNN(embedding_dim, (3, 3))
        self.norm3 = tf.keras.layers.BatchNormalization()
        self.fc = tf.keras.layers.Dense(embedding_dim, activation='relu')

    def call(self, x):
        # x shape after cnn1 == (batch_size, 62, 62, embedding_dim // 16)
        x = self.res_cnn1(x)
        x = self.norm1(x)
        x = tf.nn.relu(x)
        # x shape after pool1 == (batch_size, 31, 31, embedding_dim // 16)
        x = self.pool1(x)
        
        # x shape after cnn2 == (batch_size, 29, 29, embedding_dim // 4)
        x = self.res_cnn2(x)
        x = self.norm2(x)
        x = tf.nn.relu(x)
        # x shape after pool2 == (batch_size, 14, 14, embedding_dim // 4)
        x = self.pool2(x)
        
        # x shape after cnn3 == (batch_size, 12, 12, embedding_dim)
        x = self.res_cnn3(x)
        x = self.norm3(x)
        x = tf.nn.relu(x)
        # reshape from (batch_size, 12, 12, 128) to (batch_size, 144, embedding_dim)
        x = tf.reshape(x, [x.shape[0], -1, x.shape[-1]])
        # x shape after fc == (batch_size, 144, embedding_dim)
        x = self.fc(x)
        return x


class Bahdanau_Attention(tf.keras.Model):
    def __init__(self, attention_dim):
        super(Bahdanau_Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(attention_dim)
        self.W2 = tf.keras.layers.Dense(attention_dim)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_Encoder output) shape == (batch_size, 36, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 81, attention_dim)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 36, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, embedding_dim)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Simple_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, max_length, hidden_size, vocab_size):
        super(Simple_Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.attention = Bahdanau_Attention(hidden_size)
        self.fc1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        
    def call(self, feature, position):
        # y shape (batch_size, hidden_size)
        y = self.embedding(position)
        # x shape (batch_size, embedding_dim)
        x, w = self.attention(feature, y)
        # x shape (batch_size, hidden_size)
        x = self.fc1(x)
        # x shape (batch_size, vocab_size)
        x = self.fc2(x)
        return x, w


class Length_Decoder(tf.keras.Model):
    def __init__(self, max_length):
        super(Length_Decoder, self).__init__()
        self.fc1 = tf.keras.layers.Dense(max_length * 16, activation='relu')
        self.fc2 = tf.keras.layers.Dense(max_length * 4, activation='relu')
        self.fc3 = tf.keras.layers.Dense(max_length)
        
    def call(self, feature):
        x = tf.reshape(feature, (feature.shape[0], -1))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # shape = (batch_size, max_length)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, hidden_size, vocab_size, max_length):
        super(RNN_Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru1 = tf.keras.layers.GRU(self.hidden_size, return_sequences=True,
                                        return_state=True, recurrent_initializer='glorot_uniform')
        self.gru2 = tf.keras.layers.GRU(self.hidden_size, return_sequences=True,
                                        return_state=True, recurrent_initializer='glorot_uniform')
        self.gru3 = tf.keras.layers.GRU(self.hidden_size, return_sequences=True,
                                        return_state=True, recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = Bahdanau_Attention(hidden_size)

    def call(self, x, l, features, hidden):
        # x is forward direction, y is beckward direction
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden[0])
        l = tf.expand_dims(tf.cast(l, 'float32'), 1)

        # x shape before is (batch_size, 1) since it is passed through one by one at a time
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # context_vector shape is (batch_size, embedding_dim)
        # x shape after concatenation == (batch_size, 1, embedding_dim + embedding_dim)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        # x shape is (batch_size, 1, hidden_size)
        # state is new hidden used in next step
        x, state1 = self.gru1(x, initial_state = hidden[0])
        x_identity = tf.identity(x)
        x = tf.concat([l, x], axis=-1)
        x, state2 = self.gru2(x, initial_state = hidden[1])
        x_identity2 = tf.identity(x)
        x, state3 = self.gru3(x + x_identity, initial_state = hidden[2])
        # x shape (batch_size, 1, max_length + hidden_size)
        x = tf.concat([l, x + x_identity2], axis=-1)
        x = tf.reshape(x, (x.shape[0], -1))
        # x shape (batch_size, hidden_size)
        x = self.fc1(x)
        # x shape (batch_size, vocab_size)
        x = self.fc2(x)

        return x, [state1, state2, state3], attention_weights

    def reset_state(self, batch_size):
        # generate new hidden layer with different batch size
        return [tf.zeros((batch_size, self.hidden_size))] * 3


# +
optimizer_step1 = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    loss_ = loss_object(real, pred)
    return tf.reduce_mean(loss_)

def accuracy_function(real, pred):
    pred_index = tf.math.argmax(pred, axis=-1)
    return tf.math.reduce_mean(tf.cast(pred_index == real, tf.float32))


# -

@tf.function
def train_step1(glyph, target):
    loss = 0; accuracy = 0
    with tf.GradientTape() as tape:
        feature = encoder(glyph)
        for i in range(1, target.shape[1]):
            position = tf.convert_to_tensor(np.repeat(i-1, target.shape[0]), dtype='int64')
            prediction, weight = decoder_step1(feature, position)
            loss += tf.reduce_mean(loss_object(target[:, i], prediction))
            accuracy += accuracy_function(target[:, i], prediction)
    trainable_variables = decoder_step1.trainable_variables + encoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer_step1.apply_gradients(zip(gradients, trainable_variables))
    return loss / (target.shape[1] - 1), accuracy / (target.shape[1] - 1)


@tf.function
def validation_step1(glyph, target):
    loss = 0; accuracy = 0
    feature = encoder(glyph)
    for i in range(1, target.shape[1]):
        position = tf.convert_to_tensor(np.repeat(i-1, target.shape[0]), dtype='int64')
        prediction, weight = decoder_step1(feature, position)
        loss += tf.reduce_mean(loss_object(target[:, i], prediction))
        accuracy += accuracy_function(target[:, i], prediction)
    return loss / (target.shape[1] - 1), accuracy / (target.shape[1] - 1)


def step1(epoch):
    start = time.time()
    total_loss = 0
    total_accuracy = 0

    for (batch, (glyph_tensor, target, length)) in enumerate(dataset):
        t_loss, accuracy = train_step1(glyph_tensor, target)
        total_loss += t_loss
        total_accuracy += accuracy
        print(f'Epoch {epoch + 1}, Train Loss {total_loss/batch:.6f}, Accuracy {total_accuracy / batch:.2%};\
 progression {batch / num_steps:.1%}, time elapsed {time.time() - start:.2f} sec', end='\r')
    
    val_loss, val_accuracy = validation_step1(validation_glyphs, validation_tokens)
   
    # storing the epoch end loss value to plot later 
    ckpt_manager_step1.save()

    print (f'Epoch {epoch+1}, Train Loss {total_loss/num_steps:.6f}, Accuracy {total_accuracy/num_steps:.2%};\
 Validation Loss {val_loss:.6f}, Accuracy {val_accuracy:.2%}; taken {time.time() - start:.2f} sec')


@tf.function
def predict(features, max_length):
    # start with 0
    dec_input = tf.convert_to_tensor([[0]]*features.shape[0], dtype='int64')
    hidden = decoder.reset_state(batch_size=features.shape[0])
    length = tf.nn.softmax(length_decoder(features), axis=-1)
    # iterate predictions, no teacher forcing here
    for i in range(max_length):
        prediction, hidden, attention_weights = decoder(tf.expand_dims(dec_input[:, i], 1), length, features, hidden)
        # we need deterministic result
        predicted_id = tf.math.argmax(prediction, axis=-1)
        dec_input = tf.concat([dec_input, tf.expand_dims(predicted_id, 1)], axis=1)
    return dec_input


@tf.function
def predict_next(features, target, length):
    hidden = decoder.reset_state(batch_size=features.shape[0])
    predictions = tf.constant(0, dtype='float32', shape=(features.shape[0], 1, VOCAB))
    for i in range(target.shape[1]-1):
        prediction, hidden, attention_weights = decoder(tf.expand_dims(target[:, i], 1), length, features, hidden)
        predictions = tf.concat([predictions, tf.expand_dims(prediction, 1)], axis=1)
    return predictions[:, 1:, :]


# +
optimizer_step2 = tf.keras.optimizers.Adam()
optimizer_length = tf.keras.optimizers.Adam()

def loss_function_step2(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    loss_ = tf.reduce_mean(loss_, axis=0)
    return tf.reduce_sum(loss_)

def accuracy_function_step2(real, pred):
    accuracy = tf.math.reduce_all(pred == real, 1)
    return tf.math.reduce_mean(tf.cast(accuracy, tf.float32))


# -

@tf.function
def train_step2(glyph_tensor, target, length, rnn_only=False):
    # use tape to auto generate gradients
    if rnn_only:
        features = encoder(glyph_tensor)
        with tf.GradientTape() as tape:
            predictions = predict_next(features, target, length)
            loss = loss_function_step2(target[:, 1:], predictions)
    else:
        with tf.GradientTape() as tape:
            features = encoder(glyph_tensor)
            predictions = predict_next(features, target, length)
            loss = loss_function_step2(target[:, 1:], predictions)
    with tf.GradientTape() as tape_length:
        length_pred = length_decoder(features)
        loss_length = loss_function(tf.math.argmax(length, axis=-1), length_pred)
    # calculate accuracy based on the code's whole string
    predictions_id = predict(features, target.shape[1]-1)
    accuracy = accuracy_function_step2(predictions_id, target)

    trainable_variables = decoder.trainable_variables
    if not rnn_only:
        trainable_variables += encoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer_step2.apply_gradients(zip(gradients, trainable_variables))
    gradients_length = tape_length.gradient(loss_length, length_decoder.trainable_variables)
    optimizer_length.apply_gradients(zip(gradients_length, length_decoder.trainable_variables))

    return loss / (target.shape[1] - 1), accuracy


@tf.function
def validation_step2(glyph_tensor, target, length):
    features = encoder(glyph_tensor)
    predictions = predict_next(features, target, length)
    loss = loss_function_step2(target[:, 1:], predictions)
    
    # calculate accuracy based on the code's whole string
    predictions_id = predict(features, target.shape[1]-1)
    accuracy = accuracy_function_step2(predictions_id, target)
    
    return loss / (target.shape[1] - 1), accuracy


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
graph_log_dir = 'logs/gradient_tape/' + current_time + '/func'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
graph_summary_writer = tf.summary.create_file_writer(graph_log_dir)


def step2(epoch, rnn_only=False):
    start = time.time()
    total_loss = 0
    total_accuracy = 0

    for (batch, (glyph_tensor, target, length)) in enumerate(dataset):
        if batch == 0:
            tf.summary.trace_on(graph=True, profiler=True)
        t_loss, accuracy = train_step2(glyph_tensor, target, length, rnn_only=rnn_only)
        if batch == 0:
            with graph_summary_writer.as_default():
                tf.summary.trace_export(name="train_trace", step=epoch, profiler_outdir=graph_log_dir)
            tf.summary.trace_off()
        total_loss += t_loss
        total_accuracy += accuracy
        print(f'Epoch {epoch + 1}, Train Loss {total_loss/batch:.6f}, Accuracy {total_accuracy/batch:.2%};\
 progression {batch / num_steps:.1%}, time elapsed {time.time() - start:.2f} sec', end='\r')
    
    val_loss, val_accuracy = validation_step2(validation_glyphs, validation_tokens, validation_lengths)
   
    # storing the epoch end loss value to plot later
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', (total_loss / num_steps), step=epoch)
        tf.summary.scalar('accuracy', (total_accuracy / num_steps), step=epoch)
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', val_loss, step=epoch)
        tf.summary.scalar('accuracy', val_accuracy, step=epoch)
    
    ckpt_manager_step2.save()

    print(f'Epoch {epoch + 1}, Train Loss {total_loss/num_steps:.6f},\
 Accuracy {total_accuracy / num_steps:.2%}; Validation Loss {val_loss:.6f},\
 Accuracy {val_accuracy:.2%}; taken {time.time() - start:.2f} sec')


# ## Training

encoder = CNN_Encoder(embedding_dim = 128)
decoder_step1 = Simple_Decoder(embedding_dim = 128, max_length = train_tokens.shape[1]-1,
                              hidden_size = 128, vocab_size = VOCAB)

length_decoder = Length_Decoder(max_length = train_lengths.shape[1])
decoder = RNN_Decoder(embedding_dim=128, hidden_size=128, max_length = train_lengths.shape[1], vocab_size=VOCAB)

# use a checkpoint to store weights
checkpoint_path_step1 = './checkpoints/train_step1'
ckpt_step1 = tf.train.Checkpoint(encoder=encoder, decoder=decoder_step1, optimizer=optimizer_step1)
ckpt_manager_step1 = tf.train.CheckpointManager(ckpt_step1, checkpoint_path_step1, max_to_keep=5)

# use a checkpoint to store weights
checkpoint_path_step2 = "./checkpoints/train_step2"
ckpt_step2 = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer_step2)
ckpt_manager_step2 = tf.train.CheckpointManager(ckpt_step2, checkpoint_path_step2, max_to_keep=5)

# +
EPOCHS_step1 = 2
num_steps = len(train_glyphs) // BATCH_SIZE

epoch_step1 = 0
if ckpt_manager_step1.latest_checkpoint:
    epoch_step1 = int(ckpt_manager_step1.latest_checkpoint.split('-')[-1])
# -

while epoch_step1 < EPOCHS_step1:
    step1(epoch_step1)
    epoch_step1 += 1

# +
EPOCHS_step2 = 2

epoch_step2 = 0
if ckpt_manager_step2.latest_checkpoint:
    epoch_step2 = int(ckpt_manager_step2.latest_checkpoint.split('-')[-1])
    ckpt_step2.restore(ckpt_manager_step2.latest_checkpoint)
# -

while epoch_step2 < EPOCHS_step2:
    step2(epoch_step2)
    epoch_step2 += 1


# ## Testing

def evaluate(word):
    test_input = []
    for char in word:
        glyph = glyphbook.draw(char)
        if glyph is not None:
            test_input.append(glyph)
        else:
            raise ValueError(f'Character {char} unsupported.')
    test_input = tf.expand_dims(test_input, -1)
    features = encoder(test_input)
    test_result = predict(features, 5)

    def decode(indexes):
        code = ''
        for i in indexes:
            if i <= 0:
                continue
            elif i >= 27:
                break
            else:
                code += chr(i + 96)
        return code

    return np.apply_along_axis(decode, 1, test_result.numpy())


evaluate('中國')
