#!/bin/bash

# train unimernet_base
torchrun --nproc_per_node=4 train.py --cfg-path configs/train/unimernet_base_encoder6666_decoder8_dim1024.yaml

# train unimernet_small
#torchrun --nproc_per_node=4 train.py --cfg-path configs/train/unimernet_small_encoder6666_decoder8_dim768.yaml

# train unimernet_tiny
#torchrun --nproc_per_node=4 train.py --cfg-path configs/train/unimernet_tiny_encoder6666_decoder8_dim768.yaml