from asyncore import write
from turtle import hideturtle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from utils import load_checkpoint, save_checkpoint, print_examples
from torch.utils.tensorboard import SummaryWriter
from get_loader import get_loader
from model import CNN2RNN

from tqdm import tqdm


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    train_loader, dataset = get_loader(
        root_folder="flickr8k/images", annotation_file="flickr8k/captions.txt", transform=transform)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = True
    save_model = True

    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 3
    learning_rate = 3e-4
    num_epochs = 100

    writer = SummaryWriter("logs")
    step = 0

    model = CNN2RNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterian = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        step = load_checkpoint(torch.load(
            "my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    for epoch in range(num_epochs):
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
            }
            save_checkpoint(checkpoint)

        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for idx, (imgs, captions) in loop:
            imgs = imgs.to(device)
            captions = captions.to(device)

            # model should learn to predict the end token <EOS>
            outputs = model(imgs, captions[:-1])
            loss = criterian(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
            loop.set_description(f"Epochs: {epoch+1}/{num_epochs}")
            loop.set_postfix({"loss": loss.item()})


if __name__ == "__main__":
    train()
