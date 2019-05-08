import midi_utils
import os
import misc
import numpy as np
import argparse


def load_songs(data_folders):

    all_samples = []
    all_lengths = []
    print("Loading Songs...")
    for folder in data_folders:
        for root, _, files in os.walk(folder):
            for file in files:
                path = root + "\\" + file
                if not (path.endswith('.mid') or path.endswith('.midi')):
                    continue
                try:
                    samples = midi_utils.midi_to_samples(path)
                except Exception as e:
                    print("ERROR ", path)
                    print(e)
                    continue
                if len(samples) < 8:
                    continue

                samples, lens = misc.generate_add_centered_transpose(samples)
                all_samples += samples
                all_lengths += lens

    assert (sum(all_lengths) == len(all_samples))
    print("Saving " + str(len(all_samples)) + " samples...")
    all_samples = np.array(all_samples, dtype=np.uint8)
    all_lengths = np.array(all_lengths, dtype=np.uint32)
    np.save('data/interim/samples.npy', all_samples)
    np.save('data/interim/lengths.npy', all_lengths)
    print("Done")


if __name__ == "__main__":
    # configure parser and parse arguments
    parser = argparse.ArgumentParser(description='Load songs and put them into a dataset.')
    parser.add_argument('--data_folder', default=["data/raw"], type=str, help='The path to the midi data', action='append')

    args = parser.parse_args()
    load_songs(args.data_folder)
