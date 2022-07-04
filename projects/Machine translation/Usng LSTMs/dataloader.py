import torch
import os
from torch.utils.data import Dataset, DataLoader
import spacy
from torch.nn.utils.rnn import pad_sequence

spacy_en = spacy.load('en_core_web_sm')
spacy_fr = spacy.load('fr_core_news_sm')


class Vocabulary:
    def __init__(self, frequency_threshold):
        self.itos = {
            0: '<PAD>',
            1: '<SOS>',
            2: '<EOS>',
            3: '<UNK>'
        }

        self.stoi = {
            '<PAD>': 0,
            '<SOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3
        }

        self.frequency_threshold = frequency_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return []

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                if frequencies[word] == self.frequency_threshold:
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi['<UNK>']
            for token in tokenized_text
        ]

    def un_numericalize(self, encoding):
        return " ".join([
            self.itos[token.data.item()] if token.data.item() in self.itos else self.itos[3]
            for token in encoding
        ])


class EngVocabulary(Vocabulary):
    def __init__(self, frequency_threshold):
        super().__init__(frequency_threshold)

    @staticmethod
    def tokenizer(text):
        return [tok.text.lower() for tok in spacy_en.tokenizer(text)]


class FrVocabulary(Vocabulary):
    def __init__(self, frequency_threshold):
        super().__init__(frequency_threshold)

    @staticmethod
    def tokenizer(text):
        return [tok.text.lower() for tok in spacy_fr.tokenizer(text)]


class CustomDataset(Dataset):
    def __init__(self, root_dir, frequency_threshold_en=2, frequency_threshold_fr=1, vocab=None):
        super(CustomDataset, self).__init__()
        self.root_dir = root_dir
        self.english = open(os.path.join(root_dir, "english.txt")).read().split("\n")[:-1]
        self.french = open(os.path.join(root_dir, "french.txt")).read().split("\n")[:-1]

        if vocab is None:
            self.vocab_en = EngVocabulary(frequency_threshold_en)
            self.vocab_fr = FrVocabulary(frequency_threshold_fr)
            self.vocab_en.build_vocabulary(self.english)
            self.vocab_fr.build_vocabulary(self.french)
        else:
            self.vocab_en = vocab[0]
            self.vocab_fr = vocab[1]

    def __len__(self):
        return len(self.english)

    def __getitem__(self, index):
        english_sentence = self.english[index]
        french_sentence = self.french[index]
        numericalized_en = [self.vocab_en.stoi['<SOS>']]
        numericalized_en += self.vocab_en.numericalize(english_sentence)
        numericalized_en.append(self.vocab_en.stoi['<EOS>'])
        numericalized_en = torch.tensor(numericalized_en)

        numericalized_fr = [self.vocab_fr.stoi['<SOS>']]
        numericalized_fr += self.vocab_fr.numericalize(french_sentence)
        numericalized_fr.append(self.vocab_fr.stoi['<EOS>'])
        numericalized_fr = torch.tensor(numericalized_fr)

        return numericalized_fr, numericalized_en


class MyCollate:
    def __init__(self, pad_idx_fr, pad_idx_en):
        self.pad_idx_fr = pad_idx_fr
        self.pad_idx_en = pad_idx_en

    def __call__(self, batch):
        fr = [item[0] for item in batch]
        en = [item[1] for item in batch]
        fr = pad_sequence(fr, padding_value=self.pad_idx_fr)
        en = pad_sequence(en, padding_value=self.pad_idx_en)
        return fr, en


def get_loader(root_dir, batch_size, shuffle, vocab=None):
    dataset = CustomDataset(root_dir, vocab=vocab)
    pad_idx_en = dataset.vocab_en.stoi['<PAD>']
    pad_idx_fr = dataset.vocab_fr.stoi['<PAD>']
    loader = DataLoader(
        dataset,
        batch_size=batch_size, shuffle=shuffle,
        collate_fn=MyCollate(pad_idx_fr, pad_idx_en),
    )
    return dataset, loader

if __name__ == "__main__":
    train_set, train_loader = get_loader("data/train", batch_size=64, shuffle=True)
    val_set, val_loader = get_loader("data/val", batch_size=64, shuffle=True,
                                     vocab=[train_set.vocab_en, train_set.vocab_fr])
