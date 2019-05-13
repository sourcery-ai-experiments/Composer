#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Miscelaneous utility functions.
"""

import os
import cv2
import numpy as np


def plot_sample(file_name, sample, threshold=None):
    if threshold is not None:
        inverted = np.where(sample > threshold, 0, 1)
    else:
        inverted = 1.0 - sample
    cv2.imwrite(file_name, inverted * 255)


def plot_samples(folder, samples, threshold=None):
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(samples.shape[0]):
        plot_sample(folder + '/s' + str(i) + '.png', samples[i], threshold)


def pad_songs(y, y_lens, max_len):
    y_shape = (y_lens.shape[0], max_len) + y.shape[1:]
    y_train = np.zeros(y_shape, dtype=np.float32)
    cur_ix = 0
    for i in range(y_lens.shape[0]):
        end_ix = cur_ix + y_lens[i]
        for j in range(max_len):
            k = j % (end_ix - cur_ix)
            y_train[i, j] = y[cur_ix + k]
        cur_ix = end_ix
    assert (end_ix == y.shape[0])
    return y_train


def sample_to_pattern(sample, ix, size):
    num_pats = 0
    pat_types = {}
    pat_list = []
    num_samples = len(sample) if type(sample) is list else sample.shape[0]
    for i in range(size):
        j = (ix + i) % num_samples
        measure = sample[j].tobytes()
        if measure not in pat_types:
            pat_types[measure] = num_pats
            num_pats += 1
        pat_list.append(pat_types[measure])
    return str(pat_list), pat_types


def embed_samples(samples):
    note_dict = {}
    n, m, p = samples.shape
    samples.flags.writeable = False
    e_samples = np.empty(samples.shape[:2], dtype=np.int32)
    for i in range(n):
        for j in range(m):
            note = samples[i, j].data
            if note not in note_dict:
                note_dict[note] = len(note_dict)
            e_samples[i, j] = note_dict[note]
    samples.flags.writeable = True
    lookup = np.empty((len(note_dict), p), dtype=np.float32)
    for k in note_dict:
        lookup[note_dict[k]] = k
    return e_samples, note_dict, lookup


def e_to_samples(e_samples, lookup):
    samples = np.empty(e_samples.shape + lookup.shape[-1:], dtype=np.float32)
    n, m = e_samples.shape
    for i in range(n):
        for j in range(m):
            samples[i, j] = lookup[e_samples[i, j]]
    return samples