import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataloader import get_loader
from model import Encoder, Decoder, Seq2Seq


train_set, train_loader = get_loader("data/train", batch_size=64, shuffle=True)
val_set, val_loader = get_loader("data/val", batch_size=64, shuffle=True,
                                     vocab=[train_set.vocab_en, train_set.vocab_fr])

num_epochs = 100
learning_rate = 0.001
batch_size = 64

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

writer = SummaryWriter("logs")

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, encoder_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size,
                      hidden_size, output_size, num_layers, decoder_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net, len(train_set.vocab_en), device).to(device)
pad_idx = train_set.vocab_en.stoi["<PAD>"]

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

total_val_loss = 0
for epoch in range(1, num_epochs+1):
    train_loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    train_loss = 0
    model.train()
    for batch_idx, (french, english) in train_loop:
        french = french.to(device)
        english = english.to(device)
        output = model(french, english)
        output = output[1:].reshape(-1, output.shape[2])
        english = english[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, english)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        train_loss += loss.data.item()
        train_loop.set_description(f"Epoch: {epoch}/{num_epochs}")
        train_loop.set_postfix({"batch_loss": loss.data.item(), "train_loss":train_loss, "val_loss": total_val_loss})
        # ---- validation-----
    model.eval()
    with torch.inference_mode():
        val_loss = 0
        val_loop = tqdm(val_loader, total=len(val_loader), leave=False)
        for french, english in val_loop:
            french = french.to(device)
            english = english.to(device)
            output = model(french, english)
            output = output[1:].reshape(-1, output.shape[2])
            english = english[1:].reshape(-1)
            loss = criterion(output, english).data.item()
            val_loss += loss

            val_loop.set_description("Validating")
            val_loop.set_postfix({'val loss': loss})
            writer.add_scalar("Training loss", train_loss, global_step=epoch)
        total_val_loss = val_loss
