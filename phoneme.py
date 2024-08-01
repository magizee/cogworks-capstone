from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import numpy as np

# Load the Wav2Vec2 processor and model
def initialize_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

arphabet_to_ipa = {
    'aa': 'ɑ',
    'ae': 'æ',
    'ah':'ʌ',
    'ao':'ɔ',
    'aw':'W',
    'ax':'ə',
    'axr':'ɚ',
    'ay':'Y',
    'eh':'ɛ',
    'er':'ɝ',
    'ey':'e',
    'ih':'ɪ',
    'ix':'ɨ',
    'iy':'i',
    'ow':'o',
    'oy':'O',
    'uh':'ʊ',
    'uw':'u',
    'ux':'ʉ',
    'b':'b',
    'ch':'C',
    'd':'d',
    'dh':'ð',
    'dx':'ɾ',
    'el':'l̩',
    'em':'m̩',
    'en':'n̩',
    'f':'f',
    'g':'g',
    'hh':'h',
    'h':'h',
    'jh':'J',
    'k':'k',
    'l':'l',    
    'm':'m',    
    'n':'n',    
    'ng':'ŋ',    
    'nx':'ɾ̃',    
    'p':'p',    
    'q':'ʔ',    
    'r':'ɹ',    
    's':'s',    
    'sh':'ʃ',    
    't':'t',    
    'th':'θ',    
    'v':'v',    
    'w':'w',    
    'wh':'ʍ',    
    'y':'j',    
    'z':'z',    
    'zh':'ʒ',    
    'ax-h':'ə̥',    
    'bcl':'b̚',    
    'dcl':'d̚',    
    'eng':'ŋ̍',    
    'gcl':'ɡ̚',    
    'hv':'ɦ',    
    'kcl':'k̚',    
    'pcl':'p̚',    
    'tcl':'t̚',
    'epi':'S', 
    'pau':'P',   
}

arphabet = list(arphabet_to_ipa.keys())
arphabet_to_num = {a: i for i, a in enumerate(arphabet)}

def read_phoneme_file(phn_path):
    phonetic_segments = []
    with open(phn_path, 'r') as file:
        for line in file:
            line = line.strip()
            _, _, label = line.split()
            phonetic_segments.append(label)
    return phonetic_segments

def del_unnecessary_phonetic(x):
    return x[1:-1] # remove h#

# Convert phonetic code to IPA represented in one character
def convert_to_ipa(phoneme): # phoneme is the phonetic list
    phoneme_abstract = ''.join([arphabet_to_ipa[code] for code in phoneme])
    return phoneme_abstract

def convert_to_num(phoneme):
    return [arphabet_to_num[p] for p in phoneme]

def speech_file_to_array(wav_path):
    samples, sample_rate = librosa.load(wav_path)
    # Ensure the audio is at the correct sample rate
    if sample_rate != 16000:
        audio_input = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)

    return audio_input / np.abs(audio_input).max() #normalizing audio to [-1, 1]

def phoneme_abstraction(phn_path):
    #phonetic_segments = []
    phonetic_segments = read_phoneme_file(phn_path)
    phoneme = del_unnecessary_phonetic(phonetic_segments)
    return convert_to_num(phoneme)

def sentence_being_read(txt_path):
    with open(txt_path, 'r') as file:
        for line in file:
            line = line.strip()
    return line[8:]

import torch

def get_needed_data(wav_path, txt_path, phn_path):
    audio_arr = speech_file_to_array(wav_path)
    token = (processor(audio_arr, sampling_rate=16000, padding=True, return_tensors="pt").input_values).to(device)
    phoneme_abs = phoneme_abstraction(phn_path)
    phoneme_token = torch.tensor(phoneme_abs, dtype=torch.int32)
    """phoneme = read_phoneme_file(phn_path)
    phoneme_token = (processor.tokenizer(phoneme, return_tensors="pt", padding=True, truncation=True).input_ids).to(device)"""

    """phonetic_segments = read_phoneme_file(phn_path)
    phoneme = del_unnecessary_phonetic(phonetic_segments)
    phoneme_converted = convert_to_ipa(phoneme)
    phoneme_token = (processor.tokenizer(phoneme_converted, return_tensors="pt").input_ids).to(device)"""

    line = sentence_being_read(txt_path)
    token_len = torch.tensor([len(token)], dtype=torch.int32)
    phoneme_abs_len = torch.tensor([len(phoneme_token)], dtype=torch.int32)
    #return line, audio_arr, token, phenome_abs, len(token), len(phenome_abs)
    return line, audio_arr, token, phoneme_token, token_len, phoneme_abs_len