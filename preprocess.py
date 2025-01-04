import os
import music21 as m21
import json
import tensorflow.keras as keras
import numpy as np

# Settings for environment setup for MuseScore
env = m21.environment.Environment()
env['musicxmlPath'] = r"D:\Program Files\MuseScore 4\bin\MuseScore4.exe"
env['musescoreDirectPNGPath'] = r"D:\Program Files\MuseScore 4\bin\MuseScore4.exe"

# Standard variables
DATASET_PATH = 'deutschl/erk'
ACCEPTED_DURATION = [0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4]
SAVE_DIR = 'dataset'
SINGLE_FILE_DATASET = 'file_dataset'
SEQUENCE_LENGTH = 64
MAPPING_PATH = 'mapping.json'

# Loading the song
def load_song(dataset_path):
    songs = []
    # Go through all the files in the dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".krn"):
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
                
    return songs

# Filtering out the songs that have non-acceptable duration
def has_acceptable_durations(song, accepted_durations):
    for note in song.flatten().notesAndRests:
        if note.duration.quarterLength not in accepted_durations:
            return False
    return True


def transpose(song):
    
    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]
    
    # Estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze('key')
    
    # get interval from transposition E.g., Bmaj -> Cmaj
    if key.mode == 'major':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('C'))
    elif key.mode == 'minor':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('A'))
    
    # Transpose song by calculated interval
    transposed_song = song.transpose(interval)
    
    return transposed_song
    

def encode_song(song, time_step=0.25):
    
    # p=60, d=1.0 -> [60, "_", "_", "_"]
    # So we need to encode the notes and rests
    
    encoded_song = []
    
    for event in song.flatten().notesAndRests:
    
        # handling notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # 60
        # handling rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        
        # Convert the note/rest into time series representation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
        
    # Cast the encoded song to a string
    encoded_song = " ".join(map(str, encoded_song))
    
    return encoded_song



# Preprocessing the data
def preprocess_data(dataset_path):
    
    '''Load the folk songs'''
    print("\nLoading songs...")
    songs = load_song(dataset_path)
    print(f"Loaded {len(songs)} songs.")
    
    # For MuseScore graph
    # song = songs[0]
    # song.show()
    
    print("Starting preprocessing....")
    for i, song in enumerate(songs):    
        
        '''Filter out the songs that have non-acceptable duration'''
        if not has_acceptable_durations(song, ACCEPTED_DURATION):
            # print(f"Song has non-acceptable duration, skipping it...")
            continue

        '''Transpose song to Cmaj/Amin'''
        # print("Starting the transposition...")
        song = transpose(song)
        
        '''Encode songs with music time series representation'''
        # print("Encoding the song...")
        encoded_song = encode_song(song)
        
        '''Save song to text file'''
        # print("Saving the song...")
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as file:
            file.write(encoded_song)
            
        # print(f"Song {i} saved at {save_path}")
    
    print("\nPreprocessing finished.")

def load(file_path):
    with open(file_path, "r") as file:
        song = file.read()
    return song


def create_single_file_datasets(dataset_path, file_dataset_path, sequence_length):
    
    print("starting create_single_file_datasets")
    
    new_song_delimiter = "/ "*sequence_length
    
    songs = ""
    
    '''Load encoded songs and delimiter'''
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs += song + " " + new_song_delimiter
    
    songs = songs[:-1]
    
    '''Save sttring that contain all the dataset'''
    with open(file_dataset_path, "w") as file:
        file.write(songs)
        
    print("Done with create_single_file_datasets")

    return songs

def create_mapping(songs, mapping_path):
    
    print("Starting create_mapping")
    
    mappings = {}
    
    '''Identify the vocabulary'''
    songs = songs.split()
    vocabulary = list(set(songs))
    
    '''Create the mappings'''
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i
    
    '''Save the vocab to json file'''
    with open(mapping_path, "w") as file:
        json.dump(mappings, file, indent=4)
    
    print("Done with create_mapping")
    
    
def convert_songs_to_int(songs):
    
    int_songs = []
    
    '''Load mappings'''
    with open(MAPPING_PATH, "r") as file:
        mappings = json.load(file)
    
    '''cast songs string to a list'''
    songs = songs.split()
    
    '''map song to int'''
    for symbol in songs:
        int_songs.append(mappings[symbol])
    
    return int_songs
    
def generate_training_sequences(sequence_length):
    print("Starting generate_training_sequences")
    # [11, 12, 13, 14 ...] -> input: [11, 12], target: [13], i:[11, 12, 13], t:[14]
    
    '''load the songs and map them to int'''
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)
        
    '''generate the training sequences'''
    # how many seq should/can we generate
    # 100 symbols, 64 seq len, 100-36 no of seq
    
    inputs = []
    targets = []
    
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])
    
    '''One hot encode the sequences'''
    # input: (no of seq, seq len)
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)
    
    print("dimension of input and target are: ", inputs.shape, targets.shape)
    return inputs, targets
    
def main():
    preprocess_data(DATASET_PATH)
    songs = create_single_file_datasets(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    


if __name__ == "__main__":    
    main()