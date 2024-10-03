import librosa
import numpy as np

def extract_needed_data(path):
    audio_arr = speech_file_to_array(path+'.WAV')
    line = sentence_being_read(path+'.TXT')
    phoneme_list = phoneme_abstraction(path+'.PHN')
    return audio_arr, line, phoneme_list

def speech_file_to_array(wav_path):
    samples, sample_rate = librosa.load(wav_path)
    if sample_rate != 16000: # Ensures the audio is at the correct sample rate
        audio_input = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
    return audio_input / np.abs(audio_input).max() # normalizing audio to [-1, 1]

def sentence_being_read(txt_path):
    with open(txt_path, 'r') as file:
        for line in file:
            line = line.strip()
    return line[8:]

def phoneme_abstraction(phn_path):
    phonetic_segments = read_phoneme_file(phn_path)
    phoneme = del_unnecessary_phonetic(phonetic_segments)
    return '|'.join(phoneme)

def read_phoneme_file(phn_path):
    phonetic_segments = []
    with open(phn_path, 'r') as file:
        for line in file:
            line = line.strip()
            _, _, label = line.split()
            phonetic_segments.append(label)
    return phonetic_segments

def del_unnecessary_phonetic(x):
    return x[1:-1] # removing h#