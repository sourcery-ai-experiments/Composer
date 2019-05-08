# Composer
Generates video game music using neural networks.

https://youtu.be/UWxfnNXlVy8

## How to run:
* Find some dataset to train on. More info on where to find datasets are in data/raw/README.md.
* Run load_songs.py. This will load all midi files from your midi files folder into data/interim/samples.npy & lengths.npy.
  You can point the script to a location using the --data_folder flag one or multiple times.
* Run train.py. This will train your network and store the progress from time to time (EPOCHS_TO_SAVE to change storing frequency).
  Only stops if you interrupt it or after 2000 epochs.
* Run live_edit.py --model "e1/" where "e1/" indicates the folder the stored model is in.

Have fun!
