# Music Generation using `LSTM Tensorflow`
This is `generative model` which when given some initial seed (starting note and rests) generate a whole melody music

## We are going to use
1. `Music21`
2. `LSTM-TensorFlow`

> The song data is preprocessed using python library `music21`
> Then we have transpose the songs like converting them in C major and A minor
> Then we have used LSTM layer to make the model
> We have to give an initial seed to generate the music

### E.g. of seed
`64 _ _ 62 _ r _ 60 _.....`
Here r is rest and numbers are notes

### E.g. of generated melody
![image](https://github.com/user-attachments/assets/a0cd9f44-fb7f-4f44-8c72-2fcfde89d7fa)
