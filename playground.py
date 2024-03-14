import gradio as gr
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def compute_spectrogram(aud):
    # sr=None声音保持原采样频率， mono=False声音保持原通道数
    #data, fs = librosa.load(audio_clip, sr=None, mono=False)
    data = aud[1]
    fs=16000
    L = len(data)
    print('Time:', L / fs)
    #0.025s
    framelength = 0.025
    #NFFT点数=0.025*fs
    framesize = int(framelength * fs)
    print("NFFT:", framesize)
    #画语谱图
    plt.specgram(data, NFFT=framesize, Fs=fs, window=np.hanning(M=framesize))
    plt.ylabel('Frequency')
    plt.xlabel('Time(s)')
    plt.title('Spectrogram')

    return plt

audio_interface = gr.Interface(fn=compute_spectrogram, inputs="audio", outputs="plot", title="Audio Spectrogram")
audio_interface.launch(server_name='127.0.0.1', 
                server_port=6688,
                share=True)