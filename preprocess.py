import os
import music21 as m21


# Loading the song
def load_song(dataset_path):
    pass

    # Go through all the files in the dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".krn"):
                song = m21.converter.parse(os.path.join(path, file))
                


# Preprocessing the data
def preprocess_data(dataset_path):
    print("hello world")
    pass

    # Load the folk songs
    
    # Filter out the songs that have non-acceptable duration
    
    # Transpose song to Cmaj/Amin
    
    # Encode songs with music time series representation
    
    # Save song to text file
    