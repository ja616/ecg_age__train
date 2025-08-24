
import os
import numpy as np
import matplotlib.pyplot as plt
import random

def plot_neat_ecgs(real_ecgs, real_ages, generated_ecgs, gen_ages, match_indices):
    leads_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    for gen_idx, real_idx in match_indices:
        fig, axs = plt.subplots(12, 2, figsize=(18, 15), sharex=True)
        fig.suptitle(
            f"Same Age Comparison: {real_ages[real_idx]:.1f} years\n(Left: Real PTBXL, Right: Generated)",
            fontsize=18,
            weight='bold'
        )
        for lead in range(12):
            # Real ECG
            ax_real = axs[lead, 0]
            ax_real.set_facecolor('#fff2f2')  # light pink ECG paper color
            ax_real.grid(which='major', color='#ffb3b3', linestyle='-', linewidth=0.8, alpha=0.9)
            ax_real.grid(which='minor', color='#ffc9c9', linestyle=':', linewidth=0.5, alpha=0.6)
            ax_real.minorticks_on()
            ax_real.plot(real_ecgs[real_idx, lead], color='black', linewidth=2.5)
            ax_real.set_ylim(-0.6, 0.6)
            ax_real.set_yticks([-0.5, 0, 0.5])
            ax_real.set_ylabel('mV', fontsize=10)
            ax_real.text(0.98, 0.85, leads_names[lead], ha='right', va='center',
                         transform=ax_real.transAxes, fontsize=12, weight='bold',
                         bbox=dict(facecolor='white', edgecolor='none', pad=0.3))
            ax_real.tick_params(axis='x', labelbottom=False if lead < 11 else True)

            # Generated ECG
            ax_gen = axs[lead, 1]
            ax_gen.set_facecolor('#fff2f2')
            ax_gen.grid(which='major', color='#ffb3b3', linestyle='-', linewidth=0.8, alpha=0.9)
            ax_gen.grid(which='minor', color='#ffc9c9', linestyle=':', linewidth=0.5, alpha=0.6)
            ax_gen.minorticks_on()
            ax_gen.plot(generated_ecgs[gen_idx, lead], color='red', linewidth=2.5)
            ax_gen.set_ylim(-0.6, 0.6)
            ax_gen.set_yticks([-0.5, 0, 0.5])
            ax_gen.text(0.98, 0.85, leads_names[lead], ha='right', va='center',
                        transform=ax_gen.transAxes, fontsize=12, weight='bold',
                        bbox=dict(facecolor='white', edgecolor='none', pad=0.3))
            ax_gen.tick_params(axis='x', labelbottom=False if lead < 11 else True)

        axs[11, 0].set_xlabel("Time (samples)", fontsize=12)
        axs[11, 1].set_xlabel("Time (samples)", fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

if __name__ == "__main__":
    # Load PTB-XL real ECG and ages
    real_ecgs = np.load("C:\\Users\\aishw\\OneDrive\\Dokumen\\diffusion_models_ecg\\ptbxl_dataset\\ptbxl_numpy\\X_train.npy")  # Adjust path as needed
    real_ages = np.load("C:\\Users\\aishw\\OneDrive\\Dokumen\\diffusion_models_ecg\\ptbxl_dataset\\ptbxl_numpy\\Y_train.npy")
    real_ecgs = np.transpose(real_ecgs, (0, 2, 1))
    real_ages = real_ages.flatten()

    # Load generated samples and ages
    gen_dir = "C:\\Users\\aishw\\OneDrive\\Dokumen\\diffusion_models_ecg\\result_age_cond_random\\ch256_T200_betaT0.02"
    generated_ecgs = np.load(os.path.join(gen_dir, "0_samples.npy"))
    gen_ages = np.load(os.path.join(gen_dir, "0_labels_orig.npy")).flatten()

    print(f"Loaded {generated_ecgs.shape[0]} generated samples, ages range: {min(gen_ages)} to {max(gen_ages)}")

    n_to_plot = min(5, generated_ecgs.shape[0])
    chosen_indices = random.sample(range(generated_ecgs.shape[0]), n_to_plot)


    match_indices = []
    for gen_idx in chosen_indices:
        gen_age = gen_ages[gen_idx]
        # Find closest real sample by age
        real_idx = (np.abs(real_ages - gen_age)).argmin()
        match_indices.append((gen_idx, real_idx))

    plot_neat_ecgs(real_ecgs, real_ages, generated_ecgs, gen_ages, match_indices)
