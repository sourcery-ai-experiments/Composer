#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a autoencoder model to learn to encode songs.
"""

import random

import numpy as np
from matplotlib import pyplot as plt

import midi_utils
import misc
import models

#  Load Keras
print("Loading keras...")
import os
import keras

print("Keras version: " + keras.__version__)

from keras.models import Model, load_model
from keras.utils import plot_model
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.optimizers import Adam, RMSprop

EPOCHS_TO_SAVE = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450]
NUM_EPOCHS = 2000
LEARNING_RATE = 0.001  # learning rate
CONTINUE_TRAIN = False
GENERATE_ONLY = False

WRITE_HISTORY = True
NUM_RAND_SONGS = 10

# network params
DROPOUT_RATE = 0.1
BATCHNORM_MOMENTUM = 0.9
USE_EMBEDDING = False
USE_VAE = False
VAE_B1 = 0.02
VAE_B2 = 0.1

BATCH_SIZE = 350
MAX_WINDOWS = 16  # the maximal number of measures a song can have
LATENT_SPACE_SIZE = 120
NUM_OFFSETS = 16 if USE_EMBEDDING else 1

K.set_image_data_format('channels_first')

# Fix the random seed so that training comparisons are easier to make
np.random.seed(0)
random.seed(0)

def vae_loss(x, x_decoded_mean, z_log_sigma_sq, z_mean):
    xent_loss = binary_crossentropy(x, x_decoded_mean)
    kl_loss = VAE_B2 * K.mean(1 + z_log_sigma_sq - K.square(z_mean) - K.exp(z_log_sigma_sq), axis=None)
    return xent_loss - kl_loss


def plot_scores(scores, f_name, on_top=True):
    plt.clf()
    ax = plt.gca()
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.grid(True)
    plt.plot(scores)
    plt.ylim([0.0, 0.009])
    plt.xlabel('Epoch')
    loc = ('upper right' if on_top else 'lower right')
    plt.draw()
    plt.savefig(f_name)


def save_config(num_songs, model):
    with open('results/config.txt', 'w') as file_out:
        file_out.write('LEARNING_RATE:       ' + str(LEARNING_RATE) + '\n')
        file_out.write('BATCHNORM_MOMENTUM:  ' + str(BATCHNORM_MOMENTUM) + '\n')
        file_out.write('BATCH_SIZE:          ' + str(BATCH_SIZE) + '\n')
        file_out.write('NUM_OFFSETS:         ' + str(NUM_OFFSETS) + '\n')
        file_out.write('DROPOUT_RATE:        ' + str(DROPOUT_RATE) + '\n')
        file_out.write('num_songs:           ' + str(num_songs) + '\n')
        file_out.write('optimizer:           ' + type(model.optimizer).__name__ + '\n')


def generate_random_songs(func, write_dir, rand_vectors):
    """
    Generate random songs using random latent vectors.
    :param func:
    :param write_dir:
    :param rand_vectors:
    :return:
    """
    for i in range(rand_vectors.shape[0]):
        x_rand = rand_vectors[i:i + 1]
        y_song = func([x_rand, 0])[0]
        midi_utils.samples_to_midi(y_song[0], write_dir + 'rand' + str(i) + '.mid', 0.25)


def make_msee(enc, x_orig, y_orig, write_dir):
    """
    means, stddevs, evals, evecs
    :param enc:
    :param x_orig:
    :param y_orig:
    :param write_dir:
    :return:
    """
    if USE_EMBEDDING:
        x_enc = np.squeeze(enc.predict(x_orig))
    else:
        x_enc = np.squeeze(enc.predict(y_orig))

    x_mean = np.mean(x_enc, axis=0)
    x_stds = np.std(x_enc, axis=0)
    x_cov = np.cov((x_enc - x_mean).T)
    u, s, v = np.linalg.svd(x_cov)
    e = np.sqrt(s)

    print("Means: ", x_mean[:6])
    print("Evals: ", e[:6])

    np.save(write_dir + 'means.npy', x_mean)
    np.save(write_dir + 'stds.npy', x_stds)
    np.save(write_dir + 'evals.npy', e)
    np.save(write_dir + 'evecs.npy', v)
    return x_mean, x_stds, e, v


def generate_random_songs_normalized(encoder, x_orig, y_orig, decoder, write_dir, random_vectors):
    x_mean, x_stds, e, v = make_msee(encoder, x_orig, y_orig, write_dir)

    x_vectors = x_mean + np.dot(random_vectors * e, v)
    generate_random_songs(decoder, write_dir, x_vectors)

    title = ''
    if '/' in write_dir:
        title = 'Epoch: ' + write_dir.split('/')[-2][1:]

    plt.clf()
    e[::-1].sort()
    plt.title(title)
    plt.bar(np.arange(e.shape[0]), e, align='center')
    plt.draw()
    plt.savefig(write_dir + 'evals.png')

    plt.clf()
    plt.title(title)
    plt.bar(np.arange(e.shape[0]), x_mean, align='center')
    plt.draw()
    plt.savefig(write_dir + 'means.png')

    plt.clf()
    plt.title(title)
    plt.bar(np.arange(e.shape[0]), x_stds, align='center')
    plt.draw()
    plt.savefig(write_dir + 'stds.png')


def train():
    """
    Train model.
    :return:
    """

    # Create folders to save models into
    if not os.path.exists('results'):
        os.makedirs('results')
    if WRITE_HISTORY and not os.path.exists('results/history'):
        os.makedirs('history')

    # Load dataset into memory
    print("Loading Data...")
    if not os.path.exists('data/interim/samples.npy') or not os.path.exists('data/interim/lengths.npy'):
        print('No input data found, run load_songs.py first.')
        exit(1)

    y_samples = np.load('data/interim/samples.npy')
    y_lengths = np.load('data/interim/lengths.npy')

    samples_qty = y_samples.shape[0]
    songs_qty = y_lengths.shape[0]
    print("Loaded " + str(samples_qty) + " samples from " + str(songs_qty) + " songs.")
    print(np.sum(y_lengths))
    assert (np.sum(y_lengths) == samples_qty)

    print("Preparing song samples, padding songs...")
    x_shape = (songs_qty * NUM_OFFSETS, 1)  # for embedding
    x_orig = np.expand_dims(np.arange(x_shape[0]), axis=-1)

    y_shape = (songs_qty * NUM_OFFSETS, MAX_WINDOWS) + y_samples.shape[1:]  # (songs_qty, max number of windows, window pitch qty, window beats per measure)
    y_orig = np.zeros(y_shape, dtype=y_samples.dtype)  # prepare dataset array

    # fill in measure of songs into input windows for network
    song_start_ix = 0
    song_end_ix = y_lengths[0]
    for song_ix in range(songs_qty):
        for offset in range(NUM_OFFSETS):
            ix = song_ix * NUM_OFFSETS + offset  # calculate the index of the song with its offset
            song_end_ix = song_start_ix + y_lengths[song_ix]  # get song end ix
            for window_ix in range(MAX_WINDOWS):  # get a maximum number of measures from a song
                song_measure_ix = (window_ix + offset) % y_lengths[song_ix]  # chosen measure of song to be placed in window (modulo song length)
                y_orig[ix, window_ix] = y_samples[song_start_ix + song_measure_ix]  # move measure into window
        song_start_ix = song_end_ix  # new song start index is previous song end index
    assert (song_end_ix == samples_qty)
    x_train = np.copy(x_orig)
    y_train = np.copy(y_orig)

    # copy some song from the samples and write it to midi again
    test_ix = 0
    y_test_song = np.copy(y_train[test_ix: test_ix + 1])
    x_test_song = np.copy(x_train[test_ix: test_ix + 1])
    midi_utils.samples_to_midi(y_test_song[0], 'data/interim/gt.mid')

    #  create model
    if CONTINUE_TRAIN or GENERATE_ONLY:
        print("Loading model...")
        model = load_model('results/model.h5')
    else:
        print("Building model...")

        model = models.create_model(input_shape=y_shape[1:],
                                    latent_space_size=LATENT_SPACE_SIZE,
                                    dropout_rate=DROPOUT_RATE,
                                    max_windows=MAX_WINDOWS,
                                    batchnorm_momentum=BATCHNORM_MOMENTUM,
                                    use_vae=USE_VAE,
                                    vae_b1=VAE_B1,
                                    use_embedding=USE_EMBEDDING,
                                    embedding_input_shape=x_shape[1:],
                                    embedding_shape=x_train.shape[0])

        if USE_VAE:
            model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=vae_loss)
        else:
            model.compile(optimizer=RMSprop(lr=LEARNING_RATE), loss='binary_crossentropy')

        # plot model with graphvis if installed
        try:
            plot_model(model, to_file='results/model.png', show_shapes=True)
        except OSError as e:
            print(e)

    #  train
    print("Referencing sub-models...")
    decoder = K.function([model.get_layer('decoder').input, K.learning_phase()],
                         [model.layers[-1].output])
    encoder = Model(inputs=model.input, outputs=model.get_layer('encoder').output)

    random_vectors = np.random.normal(0.0, 1.0, (NUM_RAND_SONGS, LATENT_SPACE_SIZE))
    np.save('data/interim/rand.npy', random_vectors)

    if GENERATE_ONLY:
        print("Generating songs...")
        generate_random_songs_normalized(encoder, x_orig, y_orig, decoder, 'results/', random_vectors)
        for save_epoch in range(20):
            x_test_song = x_train[save_epoch:save_epoch + 1]
            y_song = model.predict(x_test_song, batch_size=BATCH_SIZE)[0]
            midi_utils.samples_to_midi(y_song, 'results/gt' + str(save_epoch) + '.mid')
        exit(0)

    print("Training model...")
    save_config(songs_qty, model)
    train_loss = []
    offset = 0

    for epoch in range(NUM_EPOCHS):
        if USE_EMBEDDING:
            history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1)
        else:
            song_start_ix = 0
            for song_ix in range(songs_qty):
                song_end_ix = song_start_ix + y_lengths[song_ix]
                for window_ix in range(MAX_WINDOWS):
                    song_measure_ix = (window_ix + offset) % (song_end_ix - song_start_ix)
                    y_train[song_ix, window_ix] = y_samples[song_start_ix + song_measure_ix]
                song_start_ix = song_end_ix
            assert (song_end_ix == samples_qty)
            offset += 1

            history = model.fit(y_train, y_train, batch_size=BATCH_SIZE, epochs=1)  # train model on reconstruction loss

        # store last loss
        loss = history.history["loss"][-1]
        train_loss.append(loss)
        print("Train loss: " + str(train_loss[-1]))

        if WRITE_HISTORY:
            plot_scores(train_loss, 'results/history/scores.png', True)
        else:
            plot_scores(train_loss, 'results/scores.png', True)

        save_epoch = epoch + 1
        if save_epoch in EPOCHS_TO_SAVE or (save_epoch % 100 == 0) or save_epoch == NUM_EPOCHS:
            write_dir = ''
            if WRITE_HISTORY:
                # Create folder to save models into
                write_dir += 'results/history/e' + str(save_epoch)
                if not os.path.exists(write_dir):
                    os.makedirs(write_dir)
                write_dir += '/'
                model.save('results/history/model.h5')
            else:
                model.save('results/model.h5')

            # Save output on last epoch
            if save_epoch == NUM_EPOCHS:
                model.save('results/model.h5')
                make_msee(encoder, x_orig, y_orig, 'results/')

            print("...Saved.")

            if USE_EMBEDDING:
                y_song = model.predict(x_test_song, batch_size=BATCH_SIZE)[0]
            else:
                y_song = model.predict(y_test_song, batch_size=BATCH_SIZE)[0]

            misc.plot_samples(write_dir + 'test', y_song)
            midi_utils.samples_to_midi(y_song, write_dir + 'test.mid')

            generate_random_songs_normalized(encoder, x_orig, y_orig, decoder, write_dir, random_vectors)

    print("...Done.")


if __name__ == "__main__":
    train()
