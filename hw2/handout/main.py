# from comet_ml import Experiment
from trainer import Trainer
from diffusion import Diffusion
from unet import Unet
import argparse
import wandb
import torch

# fix random seed for reproducibility
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
torch.cuda.manual_seed_all(2024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time_steps",
        default=500,
        type=int,
        help="The number of steps the scheduler takes to go from clean image to an isotropic gaussian. This is also the number of steps of diffusion.",
    )
    parser.add_argument(
        "--train_steps",
        default=50000,
        type=int,
        help="The number of iterations for training.",
    )
    parser.add_argument("--save_folder", default="./results_afhq", type=str)
    parser.add_argument("--data_path", default="./data/train/", type=str)
    parser.add_argument("--load_path", default=None, type=str)
    parser.add_argument(
        "--data_class", choices=["all", "cat", "dog", "wild"], default="cat", type=str
    )
    parser.add_argument("--image_size", default=128, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--unet_dim", default=64, type=int)
    parser.add_argument("--unet_dim_mults", nargs="+", default=[1, 2, 4, 8], type=int)
    parser.add_argument("--fid", action="store_true")
    parser.add_argument(
        "--save_and_sample_every",
        default=1000,
        type=int,
        help="The number of steps between periodically saving the model state, " + \
             "sampling example images, and optional calculating FID"
    )

    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()
    print(args)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    wandb.login(key = "26cab076bde822ebe7e9b685d498d3162d49b6a8") # Mine
    wandb.init(
        project="DDPM_AFHQ",
        config=args.__dict__,
        reinit=True,
        name=args.save_folder.split("/")[-1],
    )

    model = Unet(
        dim=args.unet_dim,
        dim_mults=args.unet_dim_mults,
    ).to(device)

    diffusion = Diffusion(
        model,
        image_size=args.image_size,
        channels=3,
        timesteps=args.time_steps,  # number of steps
    ).to(device)

    trainer = Trainer(
        diffusion,
        args.data_path,
        image_size=args.image_size,
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        train_num_steps=args.train_steps,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        results_folder=args.save_folder,
        load_path=args.load_path,
        dataset="train",
        data_class=args.data_class,
        device=device,
        save_and_sample_every=args.save_and_sample_every,
        fid=args.fid,
    )

    if args.visualize:
        if args.load_path is None:
            print("No model to visualize, Please provide a load path.")
            exit(0)
        trainer.visualize_diffusion()
    else:
        trainer.train()
