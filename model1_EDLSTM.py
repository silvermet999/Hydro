import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import utils
import main
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

from svrg import SVRG


def build_missingness_mask(data):
  if isinstance(data, pd.DataFrame):
    return data.notna().astype('float32')
  return pd.DataFrame(data).notna().astype('float32')


def fill_for_scaling(data, interp_limit=6):
  data = data.copy()
  data = data.interpolate(method='linear', limit=interp_limit, limit_direction='both')
  data = data.fillna(data.mean())
  data = data.fillna(0.0)
  return data

def get_series_for_outlier_detection(data_scaled, mask_raw, column_name):
  series = data_scaled[column_name].to_numpy()
  mask = mask_raw[column_name].to_numpy()
  return series, mask


def mad_bounds(series, mask=None, threshold=3.5):
  s = series[mask.astype(bool)] if mask is not None else series
  median = np.median(s)
  MAD = np.median(np.abs(s - median))
  if MAD == 0:
    MAD = 1e-9
  modified_z = 0.6745 * (series - median) / MAD
  lower = median - (threshold / 0.6745) * MAD
  upper = median + (threshold / 0.6745) * MAD
  is_outlier = np.abs(modified_z) > threshold
  return is_outlier, lower, upper



def make_windows(data, window_size):
  windows = np.lib.stride_tricks.sliding_window_view(data, window_size, axis=0)
  return windows.transpose(0, 2, 1)


def make_window_masks(mask, window_size):
  return make_windows(mask, window_size)

def train_valid_test_split(data, mask_seq, n_train_valid_rows):
  data_train_valid = data.iloc[:n_train_valid_rows,:]
  mask_train_valid = mask_seq.iloc[:n_train_valid_rows, :]

  data_test = data.iloc[n_train_valid_rows:,:]
  mask_test = mask_seq.iloc[n_train_valid_rows:, :]

  data_valid, data_train = train_test_split(data_train_valid, test_size=0.4, shuffle= False) # the last 60% data in the first 6 years used for training and the first 40% used for validation.
  mask_valid, mask_train = train_test_split(mask_train_valid, test_size=0.4, shuffle=False)

  return data_train.values, mask_train.values, data_valid.values, mask_valid.values, data_test.values, mask_test.values

def prepare_data(window_size=2 , n_train_valid_rows=20000,
                         valid_fraction=0.4, columns=None):

  raw = main.features
  if columns is not None:
    raw = raw[columns]

  mask_raw = build_missingness_mask(raw)
  data_filled = fill_for_scaling(raw)

  scaler = MinMaxScaler()
  scaler.fit(data_filled.iloc[:n_train_valid_rows, :])
  data_scaled = pd.DataFrame(scaler.transform(data_filled), columns=data_filled.columns)

  data_train_valid = data_scaled.iloc[:n_train_valid_rows, :].reset_index(drop=True)
  mask_train_valid = mask_raw.iloc[:n_train_valid_rows, :].reset_index(drop=True)
  data_test = data_scaled.iloc[n_train_valid_rows:, :].reset_index(drop=True)
  mask_test = mask_raw.iloc[n_train_valid_rows:, :].reset_index(drop=True)

  n_valid = int(len(data_train_valid) * valid_fraction)
  data_train = data_train_valid.iloc[:-n_valid, :].reset_index(drop=True)
  mask_train = mask_train_valid.iloc[:-n_valid, :].reset_index(drop=True)
  data_valid = data_train_valid.iloc[-n_valid:, :].reset_index(drop=True)
  mask_valid = mask_train_valid.iloc[-n_valid:, :].reset_index(drop=True)

  # def window_split(data_df, mask_df):
  #   windows = make_windows(data_df.to_numpy(dtype='float32'), window_size)
  #   mask_windows = make_window_masks(mask_df.to_numpy(dtype='float32'), window_size)
  #   return windows, mask_windows

  train_windows = make_windows(data_train.to_numpy(dtype='float32'), window_size)
  train_mask = make_window_masks(mask_train.to_numpy(dtype='float32'), window_size)
  valid_windows = make_windows(data_valid.to_numpy(dtype='float32'), window_size)
  valid_mask = make_window_masks(mask_valid.to_numpy(dtype='float32'), window_size)
  # valid_windows, valid_mask_windows = window_split(data_valid, mask_valid)
  # test_windows, test_mask_windows = window_split(data_test, mask_test)

  return {
    "train": (train_windows, train_mask),
    "valid": (valid_windows, valid_mask),
    "test": (data_test, mask_test),
    "scaler": scaler,
    "data_scaled": data_scaled,
    "mask_raw": mask_raw,
  }


# define custome loss function (you can use the simple 'mse' as well)
def nse_loss(y_true, y_pred, mask=None, eps=1e-6):
  if mask is not None:
    y_true = y_true[mask]
    y_pred = y_pred[mask]
  denom = torch.sum((y_true - torch.mean(y_true)) ** 2)
  denom = torch.clamp(denom, min=eps)
  return torch.sum((y_pred - y_true) ** 2) / denom

def robust_recon_loss(y_true, y_pred, mask=None, delta=1.0):
  if mask is not None:
    y_true = y_true[mask]
    y_pred = y_pred[mask]
  return nn.functional.huber_loss(y_pred, y_true, delta=delta)

class EDLSTM(nn.Module):
  def __init__(self,
               window_size,
               n_vars,
               encoder_hidden=128,
               decoder_hidden=256,
               dense_dims=(256, 64),
               dropout_p=0.2):
    super().__init__()

    self.window_size = window_size
    self.n_vars = n_vars

    self.encoder_lstm = nn.LSTM(input_size=n_vars, hidden_size=encoder_hidden, batch_first=True)
    self.decoder_lstm = nn.LSTM(input_size=encoder_hidden, hidden_size=decoder_hidden, batch_first=True)

    dense_layers = []
    in_dim = decoder_hidden
    for dim in dense_dims:
      dense_layers.append(nn.Linear(in_dim, dim))
      dense_layers.append(nn.ReLU())
      dense_layers.append(nn.Dropout(dropout_p))
      in_dim = dim
    self.dense_stack = nn.Sequential(*dense_layers)

    self.output_layer = nn.Linear(in_dim, n_vars)
    self.output_activation = nn.Sigmoid()


  def forward(self, x):
    _, (h, _) = self.encoder_lstm(x) #.unsqueeze(1))
    h_last = h[-1]

    decoder_input = h_last.unsqueeze(1).repeat(1, self.window_size, 1)

    decoder_out, _ = self.decoder_lstm(decoder_input)

    x = self.dense_stack(decoder_out)
    x = self.output_activation(self.output_layer(x))
    return x #.squeeze(1)



# identify KGE, NSE for evaluation
def nse(y_true, y_pred):
  return 1-np.sum((y_pred-y_true)**2)/np.sum((y_true-np.mean(y_true))**2)

def kge(y_true, y_pred):
  kge_r = np.corrcoef(y_true,y_pred)[1][0]
  kge_a = np.std(y_pred)/np.std(y_true)
  kge_b = np.mean(y_pred)/np.mean(y_true)
  return 1-np.sqrt((kge_r-1)**2+(kge_a-1)**2+(kge_b-1)**2)




# parameters

batch_size = 64
lr = 0.0005
epochs = 500

# load data
print("loading")
results = prepare_data()
x_train, mask_train = results["train"]
x_val, mask_val = results["valid"]
data_scaled = results["data_scaled"].to_numpy()
mask_raw  = results["mask_raw"].to_numpy()
scaler = results["scaler"]
dataset = utils.CustomDataset(x_train, mask_train)
dataset_val = utils.CustomDataset(x_val, mask_val)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
# train_loader = utils.dataset_function(dataset, batch_size=64, train=True)
# test_loader = utils.dataset_function(dataset, batch_size=32, train=False)
print("loaded")
model1 = EDLSTM(window_size=2, n_vars= 296).cuda()
print("model")


optimizer = SVRG(model1.parameters(), lr=lr)

# def cosine_annealing(step, total_steps, lr_max, lr_min):
#   return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
#
#
# sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cosine_annealing(step,
#                                                                                                  500 * len(
#                                                                                                    train_loader),
#                                                                                                  1,
#                                                                                                  1e-6 / lr))

# sched = ReduceLROnPlateau(optimizer, factor=0.3, patience=15, cooldown=30, min_lr=1e-8)



def training():
  model1.train()
  avg_loss=0.0
  for batch_idx, (data, mask) in enumerate(train_loader):
      data = data.to(torch.float).cuda()
      mask = mask.to(torch.bool).cuda()

      optimizer.zero_grad()
      recon_batch = model1(data)
      nseloss = nse_loss(data, recon_batch, mask=mask)
      reconloss = robust_recon_loss(data, recon_batch, mask=mask)
      loss = 0.1 * nseloss + 0.9 * reconloss
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=5.0)
      optimizer.step()
      avg_loss += loss.item()

  total_loss = avg_loss / len(train_loader)
  sys.stdout.write('\r epoch %2d %d/%d loss %.2f' %
                       (ep, batch_idx + 1, len(train_loader), total_loss))
  # sched.step(total_loss)


def validation():
  model1.eval()
  avg_loss = 0.0
  with torch.no_grad():
    for batch_idx, (data, mask) in enumerate(val_loader):
      data = data.to(torch.float).cuda()
      mask = mask.to(torch.bool).cuda()

      recon_batch = model1(data)
      nseloss = nse_loss(data, recon_batch, mask=mask)
      reconloss = robust_recon_loss(data, recon_batch, mask=mask)
      loss = 0.1 * nseloss + 0.9 * reconloss
      avg_loss += loss.item()

  total_loss = avg_loss / len(val_loader)
  return total_loss

def impute_with_model():
  model1.load_state_dict(torch.load("model0.23813101649284363.pt")) # generated without windows!!!!!
  model1.eval()

  with torch.no_grad():
    x = torch.tensor(data_scaled, dtype=torch.float).cuda()
    mask = torch.tensor(mask_raw, dtype=torch.float).cuda()
    recon = model1(x)
    imputed_scaled = torch.where(mask.bool(), x, recon)
  imputed_scaled_np = imputed_scaled.cpu().numpy()
  imputed_real = scaler.inverse_transform(imputed_scaled_np)

  imputed_df = pd.DataFrame(imputed_real, columns=main.features.columns)
  return imputed_df

for ep in range(epochs):
  training_loss = training()
  if ep % 20 == 18:
    val_loss = validation()
    print("VAL LOSS =====> ", val_loss)
    torch.save(model1.state_dict(), f"model{val_loss}.pt")

imputed_df = impute_with_model()
imputed_df.to_csv('imputed_output.csv', index=False)
