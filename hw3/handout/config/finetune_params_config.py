init_from = 'gpt2'

eval_interval = 5
eval_iters = 40
wandb_project = 'lora_finetune'
out_dir="lora-gpt-default"

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

batch_size = 1
gradient_accumulation_steps = 32
max_iters = 50

# finetune at constant LR
learning_rate = 5e-4
decay_lr = False

device = "mps"
compile = False
compute_grad_memory = True

lora_rank = 128
lora_alpha = 512
lora_dropout = 0.05