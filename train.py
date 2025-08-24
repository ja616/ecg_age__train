import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

print("Visible CUDA Devices:", torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("Device name:", torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU")

from sssd.utils.util import find_max_epoch, print_size, training_loss_label, calc_diffusion_hyperparams
from sssd.models.SSSD_ECG import SSSD_ECG
from sssd.ecg_data_pre import PTBXL_AgeDataset  # Adjust import path if needed


def train(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          batch_size,
          num_workers=4):
    
    local_path = "ch{}_T{}_betaT{}".format(model_config["res_channels"], 
                                           diffusion_config["T"], 
                                           diffusion_config["beta_T"])

    output_directory = os.path.join(output_directory, local_path)
    os.makedirs(output_directory, exist_ok=True)
    os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].to(device)

    net = SSSD_ECG(**model_config).to(device)
    print_size(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Mixed Precision (AMP) support
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        try:
            model_path = os.path.join(output_directory, f'{ckpt_iter}.pkl')
            checkpoint = torch.load(model_path, map_location='cpu')
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'Successfully loaded model at iteration {ckpt_iter}')
        except:
            ckpt_iter = -1
            print('No valid checkpoint found, starting from scratch.')
    else:
        ckpt_iter = -1
        print('Starting training from scratch.')

    train_dataset = PTBXL_AgeDataset(
        trainset_config["X_train_path"],
        trainset_config["Y_train_path"]
    )
    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    index_8 = torch.tensor([0, 2, 3, 4, 5, 6, 7, 11])

    n_iter = ckpt_iter + 1

    while n_iter <= n_iters:
        for audio, label in trainloader:
            audio = torch.index_select(audio, 1, index_8).float().to(device)
            label = label.float().to(device).unsqueeze(1)  # Shape (B,1)

            optimizer.zero_grad()
            X = (audio, label)

            # 🔹 Mixed Precision Forward + Loss
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = training_loss_label(net, nn.MSELoss(), X, diffusion_hyperparams)

            # 🔹 Backward + Optimizer step (scaled if AMP enabled)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if n_iter % iters_per_logging == 0:
                print(f"Iteration: {n_iter}\tLoss: {loss.item():.6f}")

            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = f'{n_iter}.pkl'
                torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(output_directory, checkpoint_name))
                print(f'Model checkpoint saved at iteration {n_iter}')

            n_iter += 1
            if n_iter > n_iters:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/SSSD_ECG.json', help='JSON config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    print(config)

    train_config = config["train_config"]
    trainset_config = config["trainset_config"]
    diffusion_config = config["diffusion_config"]
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)
    model_config = config['wavenet_config']

    train(**train_config)
