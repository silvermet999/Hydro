import sys

import torch
import numpy as np
import pandas as pd
from torch import optim
import utils
import main
from AE_archi import DAE

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def training(epoch, model, train_loader_in, train_loader_out, optimizer,
             strength=0.01, rho=0.01, beta=0.5, gamma_max=1.0):
    model.train()
    ce_avg, oe_avg = 0.0, 0.0

    loss_avg = 0.0
    gamma = 0.01

    for batch_idx, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):
        data = torch.cat((in_set, out_set), 0).to(torch.float).cuda()

        col_has_data = (~torch.isnan(data)).any(dim=0)
        mean = torch.where(col_has_data, torch.nanmean(data, dim=0), torch.zeros_like(data[0]))
        var = torch.where(col_has_data,
                          torch.nanmean((data - mean) ** 2, dim=0),
                          torch.ones_like(data[0]))
        std = var.sqrt().clamp_min(1e-6)
        data_scaled = (data - mean) / std

        optimizer.zero_grad()
        data_null = ~torch.isnan(data_scaled)
        data_filled = torch.nan_to_num(data_scaled, nan=0.0)

        encoder_input = torch.cat([data_filled, data_null.float()], dim=1)
        recon_batch = model(encoder_input)
        recon_loss = ((recon_batch - data_scaled) ** 2)[data_null].mean()

        n_in = len(in_set)
        oe_filled = data_filled[n_in:].detach()
        oe_mask = data_null[n_in:].float().detach()
        bias = torch.rand_like(oe_filled) * 0.0001

        for _ in range(20):
            bias.requires_grad_()
            aug_input = torch.cat([oe_filled + bias, oe_mask], dim=1)
            recon_aug = model(aug_input)
            l_sur = ((recon_aug - (oe_filled + bias)) ** 2).mean(1).mean()
            r_sur = bias.abs().mean(-1).mean()
            l_sur = l_sur - r_sur * gamma
            grads = torch.autograd.grad(l_sur, [bias])[0]
            grads = grads / (grads ** 2).sum(-1).sqrt().unsqueeze(1).clamp_min(1e-12)
            bias = bias.detach() - strength * grads.detach()
            optimizer.zero_grad()

        gamma -= beta * (rho - r_sur.detach())
        gamma = gamma.clamp(min=0.0, max=gamma_max)

        if epoch >= 0:
            oe_input = torch.cat([oe_filled + bias.detach(), oe_mask], dim=1)
        else:
            oe_input = torch.cat([oe_filled, oe_mask], dim=1)
        recon_oe = model(oe_input)

        l_oe = -((recon_oe - oe_input[:, :oe_filled.shape[1]]) ** 2).mean()

        loss = recon_loss + 0.5 * l_oe
        loss.backward()
        optimizer.step()

        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        ce_avg = ce_avg * 0.8 + float(recon_loss) * 0.2
        oe_avg = oe_avg * 0.8 + float(l_oe) * 0.2

        sys.stdout.write('\r epoch %2d %d/%d loss %.2f (ce %f, oe %f)' %
                         (epoch, batch_idx + 1, len(train_loader_in), loss_avg, ce_avg, oe_avg))
    return loss_avg, mean, std

# def validation(model, val_loader, mean, std):
#     model.eval()
#     val_loss = 0
#     for batch_idx, data in enumerate(val_loader):
#         data = data.float().to(device)
#         data_scaled = (data - mean) / std
#
#         data_null = ~torch.isnan(data_scaled)
#         data_filled = torch.nan_to_num(data_scaled, nan=0.0)
#         encoder_input = torch.cat([data_filled, data_null.float()], dim=1)
#         recon_batch = model(encoder_input)
#         batch_loss = ((recon_batch - data_scaled) ** 2)[data_null].mean()
#
#         val_loss += batch_loss.item()
#
#         if batch_idx % 100 == 0:
#             print(
#                 f'Val Epoch: {epoch} '
#                 f'[{batch_idx * len(data_scaled)}/{len(val_loader.dataset)}] '
#                 f'Loss: {batch_loss.item():.6f}'
#             )
#
#     val_loss /= len(val_loader)
#
#     print(
#         f'====> Epoch: {epoch} '
#         f'Average loss: {val_loss:.4f}'
#     )
#     return val_loss


ID_data, OOD_data = main.outlier_sc()
dataset_ID = utils.CustomDataset(ID_data.to_numpy())
dataset_OOD = utils.CustomDataset(OOD_data.to_numpy())
train_loader_in, train_loader_out = utils.dataset_function_OOD(dataset_ID, dataset_OOD, batch_size=64, train=True)
test_loader_in, test_loader_out = utils.dataset_function_OOD(dataset_ID, dataset_OOD, batch_size=32, train=False)


epochs = 100
model = DAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
for epoch in range(1, epochs + 1):
    train_loss, mean, std = training(epoch, model, train_loader_in, train_loader_out, optimizer)
    # if epoch % 10 == 9:
    #     val_loss = validation(model, test_loader_out, mean, std)
    #     torch.save(model.state_dict(), f"model_{val_loss}.pt")
