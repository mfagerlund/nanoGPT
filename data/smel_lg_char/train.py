# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out_smel_lg_char'
eval_interval = 10000
eval_iters = 400
log_interval = 100 

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'smel-lg-char'
wandb_run_name = 'smel-lg-gpt'

dataset = 'smel_lg_char'
batch_size = 32
block_size = 1024 # context of up to 1024 previous characters
gradient_accumulation_steps = 1

# https://huggingface.co/transformers/v2.2.0/pretrained_models.html

# about 34533 iterations per epoch
print(f'iters per epoch={1131585634/32768:0.4f}')


n_layer = 12#24
n_embd = 768#1024
n_head = 12#16
dropout = 0.2

learning_rate = 1e-3
max_iters = 34533*4
lr_decay_iters = 34533 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model
