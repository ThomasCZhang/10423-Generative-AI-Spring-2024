from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

class CustomDataLoader:
    def __init__(self, dataset,  tokenizer, batch_size=8):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt", padding="max_length", max_length=1024)
        
        # Format dataset with prompts and answers
       
        self.formatted_dataset = dataset.map(function = self._add_instruction_finetuning,
                                            remove_columns=dataset.column_names,
                                            load_from_cache_file=False)
        self.formatted_dataset.set_format(type='torch', columns=['instr_tuned_text'])

    def _add_instruction_finetuning(self, rec):
        #TODO: Add an "instr_tuned_text" field which modifies the "text" field of the dataset to be 
        # instruction tuned. Make sure to also include the label. 
        #Optional TODO: Convert label from 0/1 to "negative"/"positive" for better intuition during generation.  
        conv_dict = {0: "negative", 1: "positive"}

        # INSTRUCTION_TEMPLATE = "You will be provided a movie review. "\
        #     "Evaluate the sentiment of the movie review as positive or negative. "\
        #     "Respond with 'positive' or 'negative'."\
        #     "Here is the review: {text}. "\
        #     "Question: Is the review sentiment positive or negative? "\
        #     "Answer:{label}"
        
        # INSTRUCTION_TEMPLATE = "Review:fun and exciting. Tone:positive."\
        #     "Review:engaging, hilarious. Tone:positive. "\
        #     "Review:overused and boring. Tone:negative. " \
        #     "Review:bad, lame. Tone:negative. " \
        #     "Review:{text}. Tone:{label}."

        INSTRUCTION_TEMPLATE = "Guess if the review tone is positive or negative." \
            "Here is the review:{text}."\
            "Is the review's tone positive or negative?"\
            "The review tone is {label}."
        
        rec["instr_tuned_text"] = INSTRUCTION_TEMPLATE.format(text = rec['text'],
                                        label = conv_dict[rec['label']]) #TODO: Fill this
        return rec

    def _tokenize(self, examples):
        return self.tokenizer(examples["text"], truncation=True, padding=True, max_length = 1024)  # Dynamic padding will be applied later

    def collate_fn(self, batch):
        # Extract texts from the batch
        texts = [item['instr_tuned_text'] for item in batch]
        
        # Tokenize all texts in the batch
        tokenized_batch = self.tokenizer(texts, truncation=True, padding=True, 
                                         max_length=1024, return_tensors='pt')
        
        # Prepare labels: shift right, pad and append EOS token ID
        input_ids = tokenized_batch['input_ids']
        labels = input_ids[:, 1:].clone() # Shift right
        labels = torch.cat([labels, torch.full((labels.size(0), 1), self.tokenizer.pad_token_id, dtype=torch.long)], dim=1) # Append EOS
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return input_ids_padded, labels_padded
    
    def get_loader(self, shuffle=True):
        # print(self.formatted_dataset[0]['instr_tuned_text'])
        # raise Exception()
        return DataLoader(self.formatted_dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
