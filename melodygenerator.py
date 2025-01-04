import tensorflow.keras as keras
import json
import numpy as np
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH
from train import SAVE_MODEL_PATH
import music21 as m21


class MelodyGenerator:

    def __init__(self ,model_path = SAVE_MODEL_PATH):
        
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as file:
            self._mappings = json.load(file)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        '''
        seed is a peice of melody
        "64 _ 63 _ _ ...."
        '''
        
        '''Create seed with start symbol'''
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        '''map seed to integers'''
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            '''limit the seed to max_seqence_length'''
            seed = seed[-max_sequence_length:]

            '''one hot encode the seed'''
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            # shape -> (max_sequence_length, num_symbols)
            # convert into (1, max_sequence_length, num_symbols)
            onehot_seed = onehot_seed[np.newaxis, ...]

            '''Make a prediction'''
            probabilities = self.model.predict(onehot_seed)[0]
            # [0.1, 0.2, 0.1, 0.6] -> sum -> 1

            '''We will use temperature sampling'''
            output_int = self._sample_with_temperature(probabilities, temperature)

            '''Update the seed'''
            seed.append(output_int)

            '''Map int to our encoding'''
            output_symbol = [k for k,v in self._mappings.items() if v == output_int][0]
            
            '''Check whether we are at the end of the melody'''
            if output_symbol == "/":
                break

            '''update the melody'''
            melody.append(output_symbol)
        return melody
            

    
    def _sample_with_temperature(self ,probabilities, temperature):
        '''
        Here we want an index, we won't use np.argmax directly as thats rigid
        We want something more flexible

        temperature -> infinity -> This will lead to randommness which is not good
        temperature -> 0 -> This lead to same as argmax which chooses with max prob
        temperature ->1 1 -> Normal dist, we return the same
        
        '''   
        predictions = np.log(probabilities) / temperature
        # Now apply softmax
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
        # Now we get more homogenous distribution

        '''Sampling index'''
        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index

    def save_melody(self, melody, step_duration=0.25, format="midi", file_name = "kaggle/working/mel.midi"):
        
        '''Create a music21 stream'''
        stream = m21.stream.Stream()
        
        '''parse all the symbol in the melody and create note/rest objects'''
        # 60 _ _ _ r _ 62 _
        start_symbol = None
        step_counter = 1
        
        for i, symbol in enumerate(melody):

            # Handle case in which we have a note/rest
            if symbol != "_" or i+1 == len(melody):
                # Ensure we are dealing with note/rest beyinf first one
                if start_symbol is not None:

                    # calculate the quarter length duration
                    quarter_length_duration = step_duration*step_counter
                    
                    # handle rest
                    if start_symbol == 'r':
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # Reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # Handle case in which we have prolongation sign "_"
            else:
                step_counter += 1

        '''write the m21 string to midi file'''
        stream.write(format, file_name)

        
            
if __name__ == "__main__":
    mg = MelodyGenerator()
    seed = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.7)
    print(melody)
    mg.save_melody(melody)