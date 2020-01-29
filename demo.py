'''
Demo Script
'''
import pyaudio
import wave
import numpy as np
from panotti.models import *
from panotti.datautils import *
import preprocess_data


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
# SR = None
MAX_SHAPE = (8, 220148)
MELS = 96
MONO = False
PHASE = False


def send_email(contents, attachments=None):
    # receiver = 'e0267395@u.nus.edu'
    receiver = "e0267856@u.nus.edu"
    sender = "mtechke30fyp@gmail.com"
    yag = yagmail.SMTP(sender)
    yag.send(
        to=receiver,
        subject="Results of fall detection",
        contents=contents,
        attachments=attachments,
    )


def predict_one(signal, model):
    X = preprocess(signal)

    return model.predict(X, batch_size=1, verbose=False)[0]


def preprocess(signal, resample=SR, mono=MONO, max_shape=MAX_SHAPE, mels=MELS, phase=PHASE):

    sr = None
    if (resample is not None):
        sr = resample

    # signal, sr = load_audio(signal, mono=mono, sr=sr)

    # Reshape / pad so all output files have same shape
    # either the signal shape or a leading one
    shape = preprocess_data.get_canonical_shape(signal)
    if (shape != signal.shape):             # this only evals to true for mono
        signal = np.reshape(signal, shape)
    padded_signal = np.zeros(max_shape)
    use_shape = list(max_shape[:])
    use_shape[0] = min(shape[0], max_shape[0])
    use_shape[1] = min(shape[1], max_shape[1])
    #print(",  use_shape = ",use_shape)
    padded_signal[:use_shape[0], :use_shape[1]
                  ] = signal[:use_shape[0], :use_shape[1]]

    layers = make_layered_melgram(padded_signal, sr, mels=mels, phase=phase)

    return layers


if __name__ == "__main__":

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
        y_proba = predict_one(frames, model)
        answer = class_names[np.argmax(y_proba)]
        print('Predicted: ', answer)
        email_content = 'FALL DETECTED!!!'
        if answer == 'rndy' or answer == 'rndychair':
            email_content = outstr
            send_email(email_content)
    # combine & store all microphone data to output.wav file
    # We are not doing this at the moment
    # outputFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    # outputFile.setnchannels(CHANNELS)
    # outputFile.setsampwidth(mic.get_sample_size(FORMAT))
    # outputFile.setframerate(RATE)
    # outputFile.writeframes(b''.join(frames))
    # outputFile.close()
