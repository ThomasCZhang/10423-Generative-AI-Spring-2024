"""
Trains a character-level language model.
"""

import os
import sys
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
import pickle
# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    #gpt-mini settings
    C.model.n_layer=6
    C.model.n_query_head=6
    C.model.n_kv_head=6
    C.model.n_embd=192
    C.model.rope = False # toggle True or False to turn rope on and off respectively

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        C.tokenizer="default"
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        #The "tokenizer" is just a mapping of characters in the dataset to integers!
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = [self.stoi[s] for s in data]#data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        dix = self.data[idx:idx + self.config.block_size + 1]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    set_seed(config.system.seed)

    # construct the training dataset
    text = open('input.txt', 'r').read() # don't worry we won't run out of file handles
    train_dataset = CharDataset(config.data, text)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    print(config)
    model = GPT(config.model)
    
    if config.model.pretrained_folder!=None:
        assert os.path.normpath(os.path.abspath(config.model.pretrained_folder)) != os.path.normpath(os.path.abspath(config.system.work_dir)), "pretrained folder cannot be same as current folder. Change the folder name of your pretrained model or current directory using flags"
        model.load_pretrained(config.model.pretrained_folder)
    
    setup_logging(config)

    #some tests
    assert config.model.n_query_head % config.model.n_kv_head == 0, f"query_heads ({config.model.n_query_head}) must be divisible by kv_heads ({config.model.n_kv_head})"
    assert ((config.model.n_embd % config.model.n_query_head ==0) and (config.model.n_embd % config.model.n_kv_head == 0)), f"embed_dim ({config.model.n_embd}) must be divisible by query_heads ({config.model.n_query_head}) and kv_heads ({config.model.n_kv_head})"


    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    train_losses = []
    attn_times = []
    attn_mem = []
    # iteration callback
    def batch_end_callback(trainer):
        if trainer.iter_num % 1 == 0:
            train_losses.append(trainer.loss.item())
            attn_times.append(trainer.attn_times*1000)
            if trainer.device=="cuda":
                print(f"iter_dt {trainer.iter_dt:.2f}s; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f};attn_times {trainer.attn_times*1000:.2f}ms;mem_consumed {trainer.memory_consumed/(1024*1024):.2f}MB")
                attn_mem.append(trainer.memory_consumed/(1024*1024))
            else:
                print(f"iter_dt {trainer.iter_dt:.2f}s; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f};attn_times {trainer.attn_times*1000:.2f}ms;mem_consumed - not available on CPU")

        if (trainer.iter_num + 1) % 200 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                # context = "O God, O God!"
                context = "When shall we three meet again" # Macbeth
                encoded_context = [train_dataset.stoi[s] for s in context] #encoding using tokenizer
                x = torch.tensor(encoded_context, dtype=torch.long)[None,...].to(trainer.device)
                y, attn_time = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)
                y = y[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y]) #decoding using tokenizer
                print(completion)
                print(f"Attention computation took {attn_time*1000:.2f}ms to run for {config.data.block_size} seq length")
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print("saving loss and attention logs")
            data = {
                "train_losses": train_losses,
                "attention_computation_time": attn_times,
                "attention_computation_memory": attn_mem
            }
            with open(os.path.join(config.system.work_dir, 'train_logs.json'), 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
