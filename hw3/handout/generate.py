import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import datasets
import argparse

class ModelSampler:
    def __init__(self, out_dir, init_from="resume", device="cuda", max_new_tokens=5, temperature=1.0, top_k=200):
        self.out_dir = out_dir
        self.init_from = init_from
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k

        # Initialize sampling as part of __init__
        self._initialize_sampling()

    def _initialize_sampling(self):
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        dtype = 'bfloat16'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.float16, 'float16': torch.float16}[dtype]
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        self.test_dataset = datasets.load_dataset("rotten_tomatoes", split='test').shuffle(seed=42).select(range(100))
        
        if self.init_from == 'resume':
            ckpt_path = os.path.join(self.out_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            gptconf = GPTConfig(**checkpoint['model_args'])
            self.model = GPT(gptconf)
            self.model.load_state_dict(checkpoint['model'])
        elif self.init_from.startswith('gpt2'):
            self.model = GPT.from_pretrained(self.init_from, dict(dropout=0.0))
        else:
            print("Warning: Invalid Resume paramater!")
            return 
        
        self.model.eval()
        self.model.to(self.device)

        enc = tiktoken.get_encoding("gpt2")
        self.encode = lambda s: enc.encode(s, allowed_special={""})
        self.decode = lambda l: enc.decode(l)

    def get_generation(self, prompt):
        prompt_ids = self.encode(prompt)
        x = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)[None, ...]

        with torch.no_grad():
            with self.ctx:
                y = self.model.generate(x, self.max_new_tokens, temperature=self.temperature, top_k=self.top_k)
                return self.decode(y[0].tolist())

    def get_accuracy(self):
        #TODO: Iterate through the dataset and calculate the accuracy of your model! 
        # Hint: Use the `get_generation` function defined above to get the generation. 
        # Hint: Make sure to use the same INSTRUCTION_TEMPLATE as dataloader.py!
        # Note: Small models like GPT2 may have trouble generating the EOS token. Instead,
        # use this heuristic: if your training labels are "positive"/"negative" you can look 
        # for those terms in the first 10 characters of the generation, and consider a prediction 
        # as correct if it matches the target label

        conv_dict = {0: "negative", 1: "positive"}
        
        # INSTRUCTION_TEMPLATE = "You will be provided a movie review. "\
        #     "Evaluate the sentiment of the movie review as positive or negative. "\
        #     "Respond with 'positive' or 'negative'."\
        #     "Here is the review: {text}. "\
        #     "Question: Is the review sentiment positive or negative? "\
        #     "Answer:"
        
        # INSTRUCTION_TEMPLATE = "Review:fun and exciting. Tone:positive."\
        #     "Review:engaging, hilarious. Tone:positive. "\
        #     "Review:overused and boring. Tone:negative. " \
        #     "Review:bad, lame. Tone:negative. " \
        #     "Review:{text}. Tone:"
        
        INSTRUCTION_TEMPLATE = "Guess if the review tone is positive or negative." \
            "Here is the review:{text}."\
            "Is the review's tone positive or negative?"\
            "The review tone is"

        acc = 0
        pos_count = 0
        neg_count = 0
        for entry in self.test_dataset:
            tuned_text = INSTRUCTION_TEMPLATE.format(text = entry["text"])
            prediction = self.get_generation(tuned_text)
            pred = prediction[len(tuned_text):min(len(tuned_text)+20, len(prediction))]

            true_label = conv_dict[entry["label"]]
            if true_label == conv_dict[0]:
                wrong_label = conv_dict[1]
            else:
                wrong_label = conv_dict[0]

            print(true_label,"\t", pred)

            if true_label in pred and wrong_label not in pred.lower():
                acc += 1
            if "positive" in pred and "negative" not in pred.lower():
                pos_count += 1
            if "negative" in pred and "positive" not in pred.lower():
                neg_count += 1
        acc /= len(self.test_dataset)
        return acc, pos_count, neg_count
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get model accuracy from command line parameters.')
    parser.add_argument('--out_dir', type=str, help='Output directory for model checkpoints')
    parser.add_argument('--init_from', type=str, default='resume', help='Initialization method', choices=['resume', 'gpt2', "gpt2-medium", "gpt2-large", "gpt2-xl"])
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation')
    parser.add_argument('--max_new_tokens', type=int, default=5, help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for generation')
    parser.add_argument('--top_k', type=int, default=200, help='Top K tokens to sample from')
    args = parser.parse_args()
    
    sampler = ModelSampler(args.out_dir, args.init_from, args.device, args.max_new_tokens, args.temperature, args.top_k)
    accuracy, pos_counter, neg_counter = sampler.get_accuracy()
    print(f"Accuracy: {accuracy}, Positive Predictions: {pos_counter}, Negative Predictions: {neg_counter}")
