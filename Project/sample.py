import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'src'))

import torch

from dnadiffusion.data.dataloader import load_data
from dnadiffusion.metrics.metrics import generate_heatmap, kl_heatmap
from dnadiffusion.models.diffusion import Diffusion
from dnadiffusion.models.unet import UNet
from dnadiffusion.utils.sample_util import create_sample


def sample(model_path: str, num_samples: int = 1000, heatmap: bool = False):
    # Instantiating data and model
    print("Loading data")
    # encode_data = load_data(
    #     data_path="src/dnadiffusion/data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt",
    #     saved_data_path="src/dnadiffusion/data/encode_data.pkl",
    #     subset_list=[
    #         "GM12878_ENCLB441ZZZ",
    #         "hESCT0_ENCLB449ZZZ",
    #         "K562_ENCLB843GMH",
    #         "HepG2_ENCLB029COU",
    #     ],
    #     limit_total_sequences=0,
    #     num_sampling_to_compare_cells=1000,
    #     load_saved_data=True,
    # )

    encode_data = load_data(
        data_path="spinal/DorsalHornNeuron_Oligodendrocyte_042424.txt", # "spinal/astro_oligo_peaks_200bp_041924.txt",
        saved_data_path="src/dnadiffusion/data/neurons_oligo_limit50k_200bp.pkl",
        subset_list=[
            "DorsalHornNeuron",
            "Oligodendrocyte"
        ],
        limit_total_sequences=50000,
        num_sampling_to_compare_cells=1000,
        load_saved_data=True,
    )

    print("Instantiating unet")
    unet = UNet(
        dim=200,
        channels=1,
        # dim_mults=(1, 2, 4),
        dim_mults=(1,512/200,1024/200),
        resnet_block_groups=4,
    )

    print("Instantiating diffusion class")
    diffusion = Diffusion(
        unet,
        timesteps=50,
    )

    # Load checkpoint
    print("Loading checkpoint")
    checkpoint_dict = torch.load(model_path)
    diffusion.load_state_dict(checkpoint_dict["model"])

    # Send model to device
    print("Sending model to device")
    diffusion = diffusion.to("cuda")

    # Generating cell specific samples
    cell_num_list = encode_data["cell_types"]
    cell_list = list(encode_data["tag_to_numeric"].keys())

    for i in cell_num_list:
        print(f"Generating {num_samples} samples for cell {encode_data['numeric_to_tag'][i]}")
        create_sample(
            diffusion,
            conditional_numeric_to_tag=encode_data["numeric_to_tag"],
            cell_types=encode_data["cell_types"],
            number_of_samples=int(num_samples / 10),
            group_number=i,
            cond_weight_to_metric=1,
            save_timesteps=False,
            save_dataframe=True,
        )
    if heatmap:
        # Generate synthetic vs train heatmap
        motif_df = kl_heatmap(
            cell_list,
            encode_data["train_motifs_cell_specific"],
        )
        generate_heatmap(motif_df, "DNADiffusion", "Train", cell_list)

        # Generate synthetic vs test heatmap
        motif_df = kl_heatmap(
            cell_list,
            encode_data["test_motifs_cell_specific"],
        )
        generate_heatmap(motif_df, "DNADiffusion", "Test", cell_list)

        # Generate synthetic vs shuffle heatmap
        motif_df = kl_heatmap(
            cell_list,
            encode_data["shuffle_motifs_cell_specific"],
        )
        generate_heatmap(motif_df, "DNADiffusion", "Shuffle", cell_list)

        print("Finished generating heatmaps")
        return


if __name__ == "__main__":
    sample("checkpoints/epoch_10_DorsalHornNeuron_Oligodendrocyte_50klimit_firstTry.pt", heatmap = True)
