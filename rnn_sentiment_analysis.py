# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample dataset
sentences = [
    "I love machine learning",
    "Deep learning is powerful",
    "AI is the future",
    "I hate bugs",
    "Debugging is frustrating",
    "I hate programming"
]

# Labels (1 = Positive, 0 = Negative)
labels = np.array([1, 1, 1, 0, 0,0])

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

# Padding sequences
max_len = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_len, padding='post')
print(X)

# Build RNN model
model = Sequential([
    Embedding(input_dim=100, output_dim=8, input_length=max_len),
    SimpleRNN(16, activation='tanh'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X, labels, epochs=20, verbose=1)

# Test prediction
test_sentence = ["I hate coding"]
test_seq = tokenizer.texts_to_sequences(test_sentence)
test_pad = pad_sequences(test_seq, maxlen=max_len, padding='post')

prediction = model.predict(test_pad)
print("Prediction:", prediction)
