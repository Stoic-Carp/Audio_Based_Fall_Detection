import tkinter as tk

from tkinter.filedialog import askopenfilename
import h5py
import pyaudio
from keras.models import load_model
import numpy as np
from panotti.datautils import *
import preprocess_data

# recording configs
# Default channels are 8 for Matrix Creator and recording seconds are 5
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 8
RATE = 96000
RECORD_SECONDS = 5
WEIGHTS_PATH = 'weights.hdf5'
SR = 44100
# SR = None
MAX_SHAPE = (8, 220148)
MELS = 96
MONO = False
PHASE = False


def load_model_ext():
    global model, class_names
    status.set('loading model')

    model = load_model(WEIGHTS_PATH)    # load the model normally

    # --- Now load it again and look for additional useful metadata
    f = h5py.File(WEIGHTS_PATH, mode='r')

    # initialize class_names with numbers (strings) in case hdf5 file doesn't have any
    output_length = model.layers[-1].output_shape[1]
    class_names = [str(x) for x in range(output_length)]
    if 'class_names' in f.attrs:
        class_names = f.attrs.get('class_names').tolist()
        class_names = [x.decode() for x in class_names]
    f.close()
    status.set('model loaded')


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


def predict_one():
    X = preprocess(frames)
    y_proba = model.predict(X, batch_size=1, verbose=False)[0]
    answer = class_names[np.argmax(y_proba)]
    status.set(answer)


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


def open_file():
    """Open a file for editing."""
    filepath = askopenfilename(
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    if not filepath:
        return
    txt_edit.delete("1.0", tk.END)
    with open(filepath, "r") as input_file:
        text = input_file.read()
        txt_edit.insert(tk.END, text)
    window.title("Simple Text Editor - {}".format(filepath))


def record():
    global frames
    # create & configure microphone
    mic = pyaudio.PyAudio()
    stream = mic.open(format=FORMAT,
                      channels=CHANNELS,
                      rate=RATE,
                      input=True,
                      frames_per_buffer=CHUNK)

    status.set("* recording")

    # read & store microphone data per frame read
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        np_data = np.frombuffer(data, dtype=np.float16)
        frames.append(np_data)

    status.set("* done recording")

    # kill the mic and recording
    stream.stop_stream()
    stream.close()
    mic.terminate()

    frames = np.array(frames)


if __name__ == "__main__":
    window = tk.Tk()
    window.title("Fall Detection Demo")
    window.rowconfigure(0, minsize=200, weight=1)
    window.columnconfigure(1, minsize=200, weight=1)

    # text
    fr_text = tk.Frame(window, bg='white')
    status = tk.StringVar()
    heading = tk.Label(fr_text, text='Status', font=(
        "Helvetica", 16), anchor=tk.W, bg='white')

    text_display = tk.Label(fr_text, textvariable=status, bg='white')

    # buttons
    fr_buttons = tk.Frame(window, relief=tk.RAISED, bd=2)
    btn_open = tk.Button(fr_buttons, text="Open", command=open_file)
    btn_save = tk.Button(fr_buttons, text="Record Audio", command=record)
    btn_load = tk.Button(
        fr_buttons, text="Load RasberryNet", command=load_model_ext)
    btn_infer = tk.Button(fr_buttons, text='Predict', command=predict_one)
    btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
    btn_save.grid(row=1, column=0, sticky="ew", padx=5)
    btn_load.grid(row=2, column=0, sticky="ew", padx=5)
    btn_infer.grid(row=3, column=0, sticky="ew", padx=5)
    fr_buttons.grid(row=0, column=0, sticky="ns")

    fr_text.grid(row=0, column=1, sticky="nsew")
    heading.grid(row=0, column=1, sticky="ew")
    text_display.grid(row=1, column=1, sticky="nsew")

    window.mainloop()
