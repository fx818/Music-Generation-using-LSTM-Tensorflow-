from preprocess import SEQUENCE_LENGTH, generate_training_sequences
import tensorflow.keras as keras
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')



OUTPUT_UNITS = 38
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
NUM_UNITS = [256]
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"

def build_model(output_units, num_units, loss, learning_rate):
  
  # Create model architecture
  inputs = keras.layers.Input(shape=(None, output_units))
  x = keras.layers.LSTM(num_units[0])(inputs)
  x = keras.layers.Dropout(0.2)(x)
  outputs = keras.layers.Dense(output_units, activation='softmax')(x)
  model = keras.Model(inputs, outputs)

  # Compile the model
  model.compile(
    loss=loss,
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
    metrics = ['accuracy']
  )

  model.summary()

  return model


def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):

  # Generate the training sequences
  train_inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

  # build the network
  model = build_model(output_units, num_units, loss, learning_rate, )

  # train the model
  model.fit(train_inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

  # Save the model
  model.save(SAVE_MODEL_PATH)

