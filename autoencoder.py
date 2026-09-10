import torch
import numpy as np
import pandas as pd
from torch import nn, optim
import utils
import main

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(592, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 296)
        self.relu = nn.ReLU()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.fc6(h5)

    def forward(self, x):
        q = self.encode(x)
        return self.decode(q)

def training(epoch, model, train_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.float().to(device)
        mean = data.mean(dim=0)
        std = data.std(dim=0, unbiased=False)
        data_scaled = (data - mean) / std
        
        optimizer.zero_grad()
        data_null = ~torch.isnan(data_scaled)
        data_filled = torch.nan_to_num(data_scaled, nan=0.0)

        encoder_input = torch.cat([data_filled, data_null.float()], dim=1)
        recon_batch = model(encoder_input)
        loss = ((recon_batch - data_scaled) ** 2)[data_null].mean()
        loss.backward()
        train_loss += loss.item() * len(data_scaled)
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    return train_loss, mean, std

def validation(model, val_loader, mean, std):
    model.eval()
    val_loss = 0
    for batch_idx, data in enumerate(val_loader):
        data = data.float().to(device)
        data_scaled = (data - mean) / std
        
        data_null = ~torch.isnan(data_scaled)
        data_filled = torch.nan_to_num(data_scaled, nan=0.0)
        encoder_input = torch.cat([data_filled, data_null.float()], dim=1)
        recon_batch = model(encoder_input)
        batch_loss = ((recon_batch - data_scaled) ** 2)[data_null].mean()

        val_loss += batch_loss.item()

        if batch_idx % 100 == 0:
            print(
                f'Val Epoch: {epoch} '
                f'[{batch_idx * len(data_scaled)}/{len(val_loader.dataset)}] '
                f'Loss: {batch_loss.item():.6f}'
            )

    val_loss /= len(val_loader)

    print(
        f'====> Epoch: {epoch} '
        f'Average loss: {val_loss:.4f}'
    )
    return val_loss


dataset = utils.CustomDataset(main.features.to_numpy())
train_loader, val_loader = utils.dataset_function(dataset, batch_size=64, train=True)
test_loader = utils.dataset_function(dataset, batch_size=32, train=False)


epochs = 100
model = DAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
for epoch in range(1, epochs + 1):
    train_loss, mean, std = training(epoch, model, train_loader, optimizer)
    if epoch % 10 == 9:
        val_loss = validation(model, val_loader, mean, std)
        torch.save(model.state_dict(), f"model_{val_loss}.pt")
