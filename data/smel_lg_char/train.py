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
batch_size = 8
block_size = 1024 # context of up to 1024 previous characters
gradient_accumulation_steps = 4

#init_from='resume'

# https://huggingface.co/transformers/v2.2.0/pretrained_models.html

# about 80128 iterations per epoch
#train has 2,625,653,877 tokens
#val has 138,192,310 tokens
iterations_per_epoch=round(2625653877/32768)
print(f'iters per epoch={iterations_per_epoch}')


n_layer = 24#12#24
n_embd = 1024#768#1024
n_head = 16#12#16
dropout = 0.2

learning_rate = 1e-3
max_iters = iterations_per_epoch*5
lr_decay_iters = iterations_per_epoch # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model
