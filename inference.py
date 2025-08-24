import os
import argparse
import json
import numpy as np
import torch
from sssd.utils.util import find_max_epoch, print_size, sampling_label, calc_diffusion_hyperparams
from sssd.models.SSSD_ECG import SSSD_ECG

def generate_four_leads(tensor):
    leadI = tensor[:,0,:].unsqueeze(1)
    leadschest = tensor[:,1:7,:]
    leadavf = tensor[:,7,:].unsqueeze(1)
    leadII = (0.5*leadI) + leadavf
    leadIII = -(0.5*leadI) + leadavf
    leadavr = -(0.75*leadI) -(0.5*leadavf)
    leadavl = (0.75*leadI) - (0.5*leadavf)
    leads12 = torch.cat([leadI, leadII, leadschest, leadIII, leadavr, leadavl, leadavf], dim=1)
    return leads12

def generate(output_directory, num_samples, ckpt_path, ckpt_iter, ages_path, train_ages_path):
    # Prepare device and output folders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_path = "ch{}_T{}_betaT{}".format(model_config["res_channels"], diffusion_config["T"], diffusion_config["beta_T"])
    output_directory = os.path.join(output_directory, local_path)
    os.makedirs(output_directory, exist_ok=True)
    os.chmod(output_directory, 0o775)
    print("Output directory:", output_directory, flush=True)

    # Send diffusion hyperparams to device
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].to(device)

    # Model setup
    net = SSSD_ECG(**model_config).to(device)
    print_size(net)
    ckpt_path = os.path.join(ckpt_path, local_path)
    model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
    checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    print("Successfully loaded model at iteration", ckpt_iter)

    # --- Load and normalize ages for conditioning ---
    labels = np.load(ages_path)  # shape (n_samples, 1) or maybe (n_samples,)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    # Normalize with training min/max
    train_ages = np.load(train_ages_path)
    age_min = train_ages.min()
    age_max = train_ages.max()
    labels_norm = (labels - age_min) / (age_max - age_min)
    print(f"Conditioning on ages normalized using train min={age_min}, max={age_max}")

    chunk_size = num_samples
    total_samples = labels_norm.shape[0]
    chunks = [labels_norm[i:i+chunk_size] for i in range(0, total_samples, chunk_size)]

    for i, label_chunk in enumerate(chunks):
        cond = torch.from_numpy(label_chunk).float().to(device)

        # Inference
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        generated_audio = sampling_label(net, (cond.shape[0], 8, 1000), diffusion_hyperparams, cond=cond)

        generated_audio12 = generate_four_leads(generated_audio)
        end.record()
        torch.cuda.synchronize()
        elapsed = int(start.elapsed_time(end) / 1000)
        print(f"Generated {cond.shape} samples at iteration {ckpt_iter} in {elapsed} seconds")

        # Save samples and labels (save both normalized and unnormalized ages for clarity)
        samples_outfile = os.path.join(output_directory, f"{i}_samples.npy")
        labels_outfile = os.path.join(output_directory, f"{i}_labels_norm.npy")
        orig_labels_outfile = os.path.join(output_directory, f"{i}_labels_orig.npy")
        np.save(samples_outfile, generated_audio12.detach().cpu().numpy())
        np.save(labels_outfile, cond.detach().cpu().numpy())
        np.save(orig_labels_outfile, labels[i*chunk_size : i*chunk_size+cond.shape[0]])
        print(f"Saved generated samples and labels chunk {i} at iteration {ckpt_iter}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='sssd/config/config_SSSD_ECG.json', help='JSON config file')
    parser.add_argument('-ages_path', '--ages_path', type=str, default='ptbxl_dataset/ptbxl_numpy/Y_test.npy', help='Path to age labels (.npy)')
    parser.add_argument('-train_ages_path', '--train_ages_path', type=str, default='ptbxl_dataset/ptbxl_numpy/Y_train.npy', help='Path to training age labels (.npy), for normalization')
    parser.add_argument('-ckpt_path', '--ckpt_path', type=str, default='sssd_age_cond/', help='Checkpoint directory')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', type=int, default=12000, help='Checkpoint iteration')
    parser.add_argument('-n', '--num_samples', type=int, default=400, help='Samples per batch')
    parser.add_argument('-output_dir', '--output_dir', type=str, default='sssd_age_cond/', help='Output directory')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    print(config)
    gen_config = config.get('gen_config', {})
    global diffusion_config
    diffusion_config = config['diffusion_config']
    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)
    global model_config
    model_config = config['wavenet_config']

    generate(
        output_directory=args.output_dir,
        num_samples=args.num_samples,
        ckpt_path=args.ckpt_path,
        ckpt_iter=args.ckpt_iter,
        ages_path=args.ages_path,
        train_ages_path=args.train_ages_path
    )
