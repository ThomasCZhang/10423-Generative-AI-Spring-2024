import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import argparse
import wandb

img_size = (256,256)
num_labels = 3

wandb.login(key = "26cab076bde822ebe7e9b685d498d3162d49b6a8")

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class CsvImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if idx >= self.__len__(): raise IndexError()
        img_name = self.data_frame.loc[idx, "image"]
        image = Image.open(img_name).convert("RGB")  # Assuming RGB images
        label = self.data_frame.loc[idx, "label_idx"]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data(batch_size):
    transform_img = T.Compose([
        T.ToTensor(), 
        T.Resize(min(img_size[0], img_size[1]), antialias=True),  # Resize the smallest side to 256 pixels
        # T.Resize(28, antialias = True),
        T.CenterCrop(img_size),  # Center crop to 256x256
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize each color dimension
        # T.Grayscale()
        ])
    train_data = CsvImageDataset(
        csv_file='./data/img_train.csv',
        transform=transform_img,
    )
    test_data = CsvImageDataset(
        csv_file='./data/img_test.csv',
        transform=transform_img,
    )
    val_data = CsvImageDataset(
        csv_file='./data/img_val.csv',
        transform=transform_img
    )

    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)

    for X, y in train_dataloader:
        print(f"Shape of X [B, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    return train_dataloader, test_dataloader, val_dataloader

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         # First layer input size must be the dimension of the image
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(img_size[0] * img_size[1] * 3, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_labels)
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits
    
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.cv1 = nn.Conv2d(3, 16, kernel_size=(3,3)) # 256 x 256  to 254 x 254
        self.relu1 = nn.ReLU()
        self.cv2 = nn.Conv2d(16, 64, kernel_size = (5,5)) # 254 x 254 to 250 x 250
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(64*250*250, 3)
        # self.softmax = nn.Softmax()

    def forward(self, x):
        # x = self.flatten(x)
        x = self.cv1(x)
        x = self.relu1(x)
        x = self.cv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x



def train_one_epoch(dataloader, model, loss_fn, optimizer, t):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    model.train()
    n_sample = t*size
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item() / batch_size
        current = (batch + 1) * dataloader.batch_size
        if batch % 10 == 0:
            print(f"Train loss = {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        n_sample += batch_size
        wandb.log({'train_loss': loss}, step = n_sample)
        
def evaluate(dataloader, dataname, model, loss_fn, t, n_epochs):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    avg_loss, correct = 0, 0
    num_to_label = {0: "parrot", 1: "narwhal", 2: "axototl"}
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            avg_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            if t == n_epochs-1:
                for j, img in enumerate(X):
                    pred_label = num_to_label[torch.argmax(pred[j]).item()]
                    true_label = num_to_label[y[j].item()]
                    if pred_label != true_label:
                        temp_img = wandb.Image(img, mode = 'RGB', caption = f'{pred_label}/{true_label}')
                        wandb.log({f"{dataname}_{j}": temp_img})

    avg_loss /= size
    correct /= size
    wandb.log({f"{dataname}ing_loss": avg_loss,
               f"{dataname}ing_acc": correct,
               "epoch":  t+1})  
    
    print(f"{dataname} accuracy = {(100*correct):>0.1f}%, {dataname} avg loss = {avg_loss:>8f}")
    return avg_loss, correct
    
def main(n_epochs, batch_size, learning_rate):
    print(f"Using {device} device")
    train_dataloader, test_dataloader, val_dataloader = get_data(batch_size)
    
    model = NeuralNetwork().to(device)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay=0.01)
    
    # train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    run = wandb.init(
        project = 'hw0',
        name = 'errors',
        config = {
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "batch_size": batch_size
        }
    )

    for t in range(n_epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        train_one_epoch(train_dataloader, model, loss_fn, optimizer, t)
        evaluate(train_dataloader, "Train", model, loss_fn, t, n_epochs)
        evaluate(test_dataloader, "Test", model, loss_fn, t, n_epochs)
        evaluate(val_dataloader, "Val", model, loss_fn, t, n_epochs)

    print("Done!")

    # Save the model
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    # Load the model (just for the sake of example)
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("model.pth"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Image Classifier')
    parser.add_argument('--n_epochs', default=5, help='The number of training epochs', type=int)
    parser.add_argument('--batch_size', default=8, help='The batch size', type=int)
    parser.add_argument('--learning_rate', default=1e-3, help='The learning rate for the optimizer', type=float)

    args = parser.parse_args()
        
    main(args.n_epochs, args.batch_size, args.learning_rate)