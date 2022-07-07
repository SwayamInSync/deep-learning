import torch
import torch.nn as nn
import random


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedding)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedding, (hidden, cell))
        predictions = self.fc(output)
        predictions = predictions.squeeze(0)
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder,  decoder, eng_vocab_size, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.eng_vocab_size = eng_vocab_size
        self.device = device

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        outputs = torch.zeros(
            (target_len, batch_size, self.eng_vocab_size)).to(self.device)
        hidden, cell = self.encoder(source)
        x = target[0]
        print(x)
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random(
            ) < teacher_force_ratio else best_guess
        return outputs

    def predict(self, source, max_len=100):
        result = []
        result.append(1)  # index of <SOS> token
        hidden, cell = self.encoder(source)
        hidden, cell = hidden.unsqueeze(1), cell.unsqueeze(1)
        x = torch.tensor([1]).to(self.device)
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            best_guess = output.argmax()
            result.append(best_guess)
            print(best_guess.data.item())
            if best_guess.data.item() == 2:
                return torch.tensor(result).to(self.device)
        result.append(2)  # index of <EOS> token

        return torch.tensor(result).to(self.device)
