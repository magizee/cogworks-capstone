from datasets import load_dataset, load_metric, Audio, Dataset
from tqdm.auto import tqdm

import os
import numpy as np
import pandas as pd
import torchaudio
import random
import librosa
import json
import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np
import random

class TimitInterface:
    def __init__(self, data_dir='./TIMIT/data', train_csv='./TIMIT/train_data.csv', test_csv='./TIMIT/test_data.csv'):
        self.data_dir = os.path.abspath(data_dir)
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.data = {} # dictionary containing file paths for audio, text, and phonemes
        self.train = {} # split of data for training
        self.valid = {} # split of data for validation
        self.test = {} # split of data for test
        self.load_data()
        self.split_data()
        self.save_splits()
        self.preprocess_datasets()
        self.load_audio_files()
        self.build_vocab()



    def load_data(self):
        self.df_train = pd.read_csv(self.train_csv)
        self.df_test = pd.read_csv(self.test_csv)
        self.df = pd.concat([self.df_train, self.df_test]) #combine sets
        self.df = self.df[self.df['is_converted_audio'] == False] #filter data

        for idx, row in tqdm(self.df.iterrows()):
            path = row['path_from_data_dir']
            entry_id = path.split('.')[0]

            if entry_id not in self.data:
                self.data[entry_id] = {}
            if row['is_audio'] is True:
                self.data[entry_id]['audio_file'] = os.path.join(self.data_dir, path)
            elif row['is_word_file'] is True:
                self.data[entry_id]['word_file'] = os.path.join(self.data_dir, path)
            elif row['is_phonetic_file'] is True:
                self.data[entry_id]['phonetic_file'] = os.path.join(self.data_dir, path)

        
    def split_data(self):
        keys = [key for key in self.data.keys() if len(self.data[key]) == 3]
        random.Random(101).shuffle(keys)
        num_train = int(len(keys) * 0.8)
        num_valid = int(len(keys) * 0.1)
        num_test = len(keys) - num_train - num_valid
        
        self.train_keys = keys[:num_train]
        self.valid_keys = keys[num_train:num_train + num_valid]
        self.test_keys = keys[-num_test:]
        
        self.train = { key:self.data[key] for key in self.train_keys }
        self.valid = { key:self.data[key] for key in self.valid_keys }
        self.test  = { key:self.data[key] for key in self.test_keys }
    
    def save_splits(self):
        with open("custom_train.json", "w") as f:
            json.dump(self.train, f)
        with open("custom_valid.json", "w") as f:
            json.dump(self.valid, f)
        with open("custom_test.json", "w") as f:
            json.dump(self.test, f)

    
    def preprocess_datasets(self):
        def convert_to_feature_dict(data_dict):
            audio_files = []
            word_files = []
            phonetic_files = []
            for key, value in data_dict.items():
                audio_files.append(value['audio_file'])
                word_files.append(value['word_file'])
                phonetic_files.append(value['phonetic_file'])
            
            return {
                'audio_file': audio_files,
                'word_file': word_files,
                'phonetic_file': phonetic_files
            }
    
        train = convert_to_feature_dict(self.train)
        valid = convert_to_feature_dict(self.valid)
        test = convert_to_feature_dict(self.test)

        self.train_dataset = Dataset.from_dict(train) # dataset for training
        self.valid_dataset = Dataset.from_dict(valid)
        self.test_dataset = Dataset.from_dict(test) # dataset for testing

        def read_text_file(filepath):
            with open(filepath) as f:
                tokens = [line.split()[-1] for line in f]
                return " ".join(tokens)

        def prepare_text_data(item):
            item['text'] = read_text_file(item['word_file'])
            item['phonetic'] = read_text_file(item['phonetic_file'])
            return item

        self.train_dataset = (self.train_dataset
                             .map(prepare_text_data)
                             .remove_columns(["word_file", "phonetic_file"]))
        self.valid_dataset = (self.valid_dataset
                             .map(prepare_text_data)
                             .remove_columns(["word_file", "phonetic_file"]))
        self.test_dataset = (self.test_dataset
                            .map(prepare_text_data)
                            .remove_columns(["word_file", "phonetic_file"]))

                
    def load_audio_files(self):
        self.train_dataset = (self.train_dataset
                            .cast_column("audio_file", Audio(sampling_rate=16_000))
                            .rename_column('audio_file', 'audio'))
        self.valid_dataset = (self.valid_dataset
                            .cast_column("audio_file", Audio(sampling_rate=16_000))
                            .rename_column('audio_file', 'audio'))
        self.test_dataset = (self.test_dataset
                            .cast_column("audio_file", Audio(sampling_rate=16_000))
                            .rename_column('audio_file', 'audio'))

    def build_vocab(self):
        train_phonetics = [phone for x in self.train_dataset for phone in x['phonetic'].split()]
        valid_phonetics = [phone for x in self.valid_dataset for phone in x['phonetic'].split()]
        test_phonetics = [phone for x in self.test_dataset for phone in x['phonetic'].split()]

        vocab_train = list(set(train_phonetics)) + [' ']
        vocab_valid = list(set(valid_phonetics)) + [' ']
        vocab_test  = list(set(test_phonetics)) + [' ']
        vocab_list = list(set(vocab_train + vocab_valid + vocab_test))
        vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

        # make the space more intuitive to understand
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]

        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)

        with open('vocab.json', 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file)
