import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'src'))

from accelerate import Accelerator, DataLoaderConfiguration

from dnadiffusion.data.dataloader import load_data
from dnadiffusion.models.diffusion import Diffusion
from dnadiffusion.models.unet import UNet
from dnadiffusion.utils.train_util import TrainLoop

#import torch

def train():
    dataloader_config = DataLoaderConfiguration(split_batches=True)
    # accelerator = Accelerator(split_batches=True, log_with=["wandb"], mixed_precision="bf16")
    # accelerator = Accelerator(dataloader_config = dataloader_config, log_with=["wandb"], mixed_precision="bf16")
    accelerator = Accelerator(dataloader_config = dataloader_config, log_with=["wandb"], mixed_precision="fp16") # Changed to fp16
    # data = load_data(
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

    #     data = load_data(
    #     data_path="spinal/astro_oligo_peaks_200bp_041924.txt",
    #     subset_list=[
    #         "astro",
    #         "oligo"
    #     ],
    #     limit_total_sequences=2000,
    #     num_sampling_to_compare_cells=1000,
    #     load_saved_data=False,
    # )

    # create saved pickle
    # data = load_data(
    #     data_path="spinal/DorsalHornNeuron_Oligodendrocyte_042424.txt", # "spinal/astro_oligo_peaks_200bp_041924.txt",
    #     #saved_data_path="src/dnadiffusion/data/neurons_oligo_limit50k_200bp.pkl", #"src/dnadiffusion/data/encode_data_spinal_200bp.pkl",
    #     subset_list=[
    #         "DorsalHornNeuron",
    #         "Oligodendrocyte"
    #     ],
    #     limit_total_sequences=50000,
    #     num_sampling_to_compare_cells=1000,
    #     load_saved_data=False,
    # )
    data = load_data(
        data_path="spinal/DorsalHornNeuron_Oligodendrocyte_042424.txt", # "spinal/astro_oligo_peaks_200bp_041924.txt",
        saved_data_path="src/dnadiffusion/data/neurons_oligo_limit50k_200bp.pkl", #"src/dnadiffusion/data/encode_data_spinal_200bp.pkl",
        subset_list=[
            "DorsalHornNeuron",
            "Oligodendrocyte"
        ],
        limit_total_sequences=50000,
        num_sampling_to_compare_cells=1000,
        load_saved_data=True,
    )

    unet = UNet(
        dim=200,
        channels=1,
        #dim_mults=(1, 2, 4),
        dim_mults=(1, 512/200, 1024/200),
        resnet_block_groups=4,
    )

    diffusion = Diffusion(
        unet,
        timesteps=50,
    )

    # TrainLoop(
    #     data=data,
    #     model=diffusion,
    #     accelerator=accelerator,
    #     epochs=10000,
    #     log_step_show=50,
    #     sample_epoch=500,
    #     save_epoch=500,
    #     model_name="model_48k_sequences_per_group_K562_hESCT0_HepG2_GM12878_12k",
    #     image_size=200,
    #     num_sampling_to_compare_cells=1000,
    #     batch_size=960,
    # ).train_loop()

    TrainLoop(
        data=data,
        model=diffusion,
        accelerator=accelerator,
        epochs=20,
        lr=5e-6,
        log_step_show=1,
        sample_epoch=1000,
        save_epoch=10,
        model_name="DorsalHornNeuron_Oligodendrocyte_50klimit_firstTry", #"neurons_oligo_firstTry",
        image_size=200,
        num_sampling_to_compare_cells=20,
        batch_size=256,
    ).train_loop()


if __name__ == "__main__":
    train()
