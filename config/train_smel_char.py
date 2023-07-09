# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-smel-char'
eval_interval = 10000 # keep frequent because we'll overfit
eval_iters = 400
log_interval = 20 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'smel-char'
wandb_run_name = 'smel-gpt'

dataset = 'smel_char'
batch_size = 32
block_size = 1024 # context of up to 1024 previous characters
gradient_accumulation_steps = 1

# https://huggingface.co/transformers/v2.2.0/pretrained_models.html

# about 96k iterations per epoch
print(f'iters per epoch={516290294/32768:0.4f}')


# baby GPT mode
n_layer = 12#24
n_embd = 768#1024
n_head = 12#16
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 96175*8
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model
