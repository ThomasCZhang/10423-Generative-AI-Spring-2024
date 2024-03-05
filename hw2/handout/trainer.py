import os
import torch

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image

from tqdm import tqdm
from cleanfid import fid
import wandb

def MyLambdaFunc(x):
    return (x*2) - 1

def cycle(dl):
    while True:
        for data in dl:
            yield data


class Dataset(data.Dataset):
    def __init__(
        self, folder, image_size, data_class, augment=False, exts=["jpg", "jpeg", "png"]
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        if data_class == "all" or data_class == None:
            self.paths = [
                p for ext in exts for p in Path(f"{folder}").glob(f"**/*.{ext}")
            ]
        else:
            self.paths = [
                p
                for ext in exts
                for p in Path(f"{folder}/{data_class}").glob(f"*.{ext}")
            ]

        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
                    transforms.RandomCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Lambda(lambda t: (t * 2) - 1),
                    transforms.Lambda(MyLambdaFunc),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    # transforms.Lambda(lambda t: (t * 2) - 1),
                    transforms.Lambda(MyLambdaFunc),
                ]
            )
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert("RGB")
        return self.transform(img)


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        image_size=128,
        train_batch_size=32,
        train_lr=2e-5,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        save_and_sample_every=2000,
        results_folder="./results",
        load_path=None,
        dataset=None,
        shuffle=True,
        data_class=None,
        device=None,
        fid=False,
    ):
        super().__init__()
        self.model = diffusion_model

        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.data_class = data_class

        self.fid = fid

        augment_data = True if dataset == "train" else False
        self.train_folder = folder
        self.ds = Dataset(folder, image_size, data_class, augment=augment_data)
        print(f"dataset length: {len(self.ds)}, dataset class: {data_class}")

        self.dl = cycle(
            data.DataLoader(
                self.ds,
                batch_size=train_batch_size,
                shuffle=shuffle,
                pin_memory=True,
                num_workers=8,
                drop_last=True,
                persistent_workers=True # Aight lets see if this helps windows.
            )
        )

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if load_path != None:
            self.load(load_path)

    def save(self, itrs=None):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
        }
        if itrs is None:
            torch.save(data, str(self.results_folder / f"model.pt"))
        else:
            torch.save(data, str(self.results_folder / f"model_{itrs}.pt"))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path)

        self.step = data["step"]
        self.model.load_state_dict(data["model"])

    def train(self):
        start_step = self.step
        self.model.train()
        self.model.to(self.device)
        """
        Training loop
        
            1. Use wandb.log to log the loss of each step 
                This loss is the average of the loss over accumulation steps 
            2. Save the sampled images every self.save_and_sample_every steps
            3. Comopute the FID score
            4. Save the model every self.save_and_sample_every steps
        """
        milestone = 0
        for self.step in tqdm(range(start_step, self.train_num_steps), desc="steps"):
            u_loss = 0
            for i in range(self.gradient_accumulate_every):
                data_1 = next(self.dl)
                data_2 = torch.randn_like(data_1)

                data_1, data_2 = data_1.to(self.device), data_2.to(self.device)
                loss = torch.mean(self.model(data_1, data_2))
                u_loss += loss.item()
                (loss / self.gradient_accumulate_every).backward()

            # use wandb to log the loss
            wandb.log({"loss": u_loss / self.gradient_accumulate_every})

            self.opt.step()
            self.opt.zero_grad()

            if (self.step + 1) % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                save_folder = str(self.results_folder / f"sample_ddpm_{milestone}")
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                images = self.model.sample(512)
                for j in range(512):
                    utils.save_image(
                        images[j],
                        os.path.join(
                            save_folder, f"sample_{j}_{self.step}.png"
                        ),
                    )

                if self.fid:
                    # Make sure the images in the validation set are of the same size
                    self.val_folder = self.train_folder.replace("train", "val")
                    if len(os.listdir(self.val_folder)) == 0:
                        os.makedirs(self.val_folder, exist_ok=True)
                        # resize the images in the training set and save them in the validation set
                        for img_name in os.listdir(
                            os.path.join(self.train_folder, self.data_class)
                        ):
                            img_path = os.path.join(
                                self.train_folder, self.data_class, img_name
                            )
                            img = Image.open(img_path)
                            img = img.resize((self.image_size, self.image_size))
                            img.save(os.path.join(self.val_folder, img_name))

                    fid_score = fid.compute_fid(save_folder, self.val_folder)
                    wandb.log({"fid": fid_score})
                self.save()

        images = self.model.sample(self.batch_size)
        grid = utils.make_grid(images)
        utils.save_image(
            grid, os.path.join(self.results_folder, f"sample_{milestone}.png"), nrow=6
        )
        print("training completed")

    def visualize_diffusion(
        self,
    ):
        """
        Use trained model to visualize forward and backward diffusion process

        1. Sample the first batch of images from the dataloader
        2. Visualize the forward diffusion process at 0%, 25%, 50%, 75%, 99% of the total timesteps
        3. Use the last image from the forward diffusion process to visualize the backward diffusion
            process at 0%, 25%, 50%, 75%, 99% of the total timesteps
            you can use (percent * self.model.num_timesteps) to get the timesteps
        4. Save the images in wandb
        """
        self.model.eval()
        self.model.to(self.device)

        data_1 = next(self.dl)
        data_2 = torch.randn_like(data_1)

        data_1, data_2 = data_1.to(self.device), data_2.to(self.device)

        percent = [0.0, 0.25, 0.50, 0.75, 0.99]
        t_visualize = [int(i * self.model.num_timesteps) for i in percent]

        # visualize forward diffusion process
        img_forward = []
        for t in t_visualize:
            img = self.model.q_sample(
                data_1,
                torch.full(
                    (data_1.shape[0],), t, device=data_1.device, dtype=torch.long
                ),
                data_2,
            )
            img = torch.clamp(img, -1, 1)
            img = (img + 1) / 2
            img_forward.append(img)

        # visualize backward diffusion process
        img_backward = []
        for t in range(self.model.num_timesteps - 1, -1, -1):
            img = self.model.p_sample(
                img,
                torch.full((img.shape[0],), t, device=img.device, dtype=torch.long),
                t,
            )
            if t in t_visualize:
                img_vis = torch.clamp(img, -1, 1)
                img_vis = (img_vis + 1) / 2
                img_backward.append(img_vis)

        # save the images in wandb
        wandb.log({"forward_diffusion": [wandb.Image(img) for img in img_forward]})
        wandb.log({"backward_diffusion": [wandb.Image(img) for img in img_backward]})
        # ####################################################################
