# Train a tiny character-level model to learn "sort digits" completions.
# Run (from repo root):
#   python data/sort_char/prepare.py
#   python train.py config/train_sort_char.py
#   python sample.py --out_dir=out-sort-char --start="3 1 2 0 2 ->"

out_dir = 'out-sort-char'
eval_interval = 250
eval_iters = 100
log_interval = 10

# dataset lives in data/sort_char
dataset = 'sort_char'

# ---- model size (small + educational) ----
# Keep it small so it fits comfortably on an RTX 2080 (8GB).
n_layer = 4
n_head = 4
n_embd = 128
block_size = 128
dropout = 0.0  # deterministic synthetic task; regularization usually not needed

# ---- optimization ----
batch_size = 64
learning_rate = 1e-3
max_iters = 4000
lr_decay_iters = 4000
warmup_iters = 200
min_lr = 1e-4
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# ---- system / precision ----
device = 'cuda'
# RTX 2080 does NOT support bf16 well; fp16 is the usual choice.
dtype = 'float16'

# torch.compile can speed things up, but can also cause Triton/compiler issues on some setups.
compile = False

# ---- logging ----
wandb_log = False

