'''
Demo Script
'''

import pyaudio
import wave
import numpy as np
from panotti.models import *
from panotti.datautils import *


# recording configs
# Default channels are 8 for Matrix Creator and recording seconds are 5
CHUNK = 2048
FORMAT = pyaudio.paFloat32
CHANNELS = 8
RATE = 96000
RECORD_SECONDS = 5
WEIGHTS_PATH = 'weights.hdf5'
RECORD_SECONDS = 5
SR = 44100


def predict_one(signal, sr, model, expected_melgram_shape):
    X = make_layered_melgram(signal, sr, mels=96, phase=False)
    print("signal.shape, melgram_shape, sr = ", signal.shape, X.shape, sr)

    if (X.shape[1:] != expected_melgram_shape):   # resize if necessary, pad with zeros
        print('I SHOULDNT BE HERE!!!')
        Xnew = np.zeros([1]+list(expected_melgram_shape))
        min1 = min(Xnew.shape[1], X.shape[1])
        min2 = min(Xnew.shape[2], X.shape[2])
        min3 = min(Xnew.shape[3], X.shape[3])
        Xnew[0, :min1, :min2, :min3] = X[0, :min1, :min2, :min3]  # truncate
        X = Xnew
    return model.predict(X, batch_size=1, verbose=False)[0]


print('Loading model')
model, class_names = load_model_ext(WEIGHTS_PATH)
print('model loaded')

while True:
    rec = input('Record (y/n)')

    if str(rec) == 'y':

        # create & configure microphone
        mic = pyaudio.PyAudio()
        stream = mic.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=CHUNK)

        print("* recording")

        # read & store microphone data per frame read
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            np_data = np.frombuffer(data, dtype=np.float32)
            frames.append(np_data)

        print("* done recording")

        # kill the mic and recording
        stream.stop_stream()
        stream.close()
        mic.terminate()
    else:
        continue

    frames = np.array(frames)
    expected_melgram_shape = model.layers[0].input_shape[1:]
    y_proba = predict_one(frames, SR, model, expected_melgram_shape)
    answer = class_names[np.argmax(y_proba)]
    print('Predicted: ', answer)

# combine & store all microphone data to output.wav file
# We are not doing this at the moment
# outputFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# outputFile.setnchannels(CHANNELS)
# outputFile.setsampwidth(mic.get_sample_size(FORMAT))
# outputFile.setframerate(RATE)
# outputFile.writeframes(b''.join(frames))
# outputFile.close()
