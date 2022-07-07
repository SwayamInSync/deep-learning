import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataloader import get_loader
from model import Encoder, Decoder, Seq2Seq


train_set, train_loader = get_loader("data/train", batch_size=64, shuffle=True)
vocab_fr = train_set.vocab_fr
vocab_en = train_set.vocab_en

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size_encoder = len(train_set.vocab_fr)
input_size_decoder = len(train_set.vocab_en)
output_size = len(train_set.vocab_en)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 4
encoder_dropout = 0.5
decoder_dropout = 0.5

encoder_net = Encoder(input_size_encoder, encoder_embedding_size,
                      hidden_size, num_layers, encoder_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size,
                      hidden_size, output_size, num_layers, decoder_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net, len(
    train_set.vocab_en), device).to(device)
pad_idx = train_set.vocab_en.stoi["<PAD>"]

checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

french = "Deux jeunes hommes blancs sont dehors près de buissons."
num_french = [vocab_fr.stoi['<SOS>']]
num_french += vocab_fr.numericalize(french)
num_french.append(vocab_fr.stoi['<EOS>'])
num_french = torch.tensor(num_french)
num_french = num_french.to(device)

model.eval()
with torch.inference_mode():
    output = model.predict(num_french)
    print(output.shape)
    res = " ".join([vocab_en.itos[i.data.item()] for i in output])
    print(res)
