import pyaudio
import wave

# recording configs
# Default channels are 8 for Matrix Creator and recording seconds are 5 
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 8
RATE = 96000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

# Get user input for recording options
RECORD_SECONDS = int(input("Enter duration for recording in seconds: "))
WAVE_OUTPUT_FILENAME = input("Enter output filename: ") + ".wav"

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
    frames.append(data)

print("* done recording")

# kill the mic and recording
stream.stop_stream()
stream.close()
mic.terminate()

# combine & store all microphone data to output.wav file
# We are not doing this at the moment 
outputFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
outputFile.setnchannels(CHANNELS)
outputFile.setsampwidth(mic.get_sample_size(FORMAT))
outputFile.setframerate(RATE)
outputFile.writeframes(b''.join(frames))
outputFile.close()
