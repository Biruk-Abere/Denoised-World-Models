## Requirements
We assume you have access to a gpu that can run CUDA 9.2. Then, the simplest way to install all required dependencies is to create an anaconda environment by running:
```
conda env create -f environment.yml
```

For installing explicit packages:
```
conda env create -f explicit_environment.yml
```

After the instalation ends you can activate your environment with:
```
source activate dpi
```

## Instructions
To train an agent on the `cheetah run` task from image-based observations with natural background run:
```
python train.py \
        --domain_name cheetah \
        --task_name run \
        --encoder_type pixel \
        --decoder_type pixel \
        --action_repeat 4 \
        --save_video \
        --save_tb \
        --work_dir ./log \
        --seed $seed \
        --w_enc 1e-1 \
        --w_rec 1e-2 \
        --w_ub 1e-3 \
        --w_lb 1e-2 \
        --img_source video \
        --resource_files "~/path_to_training_data/train/*.mp4" \    # Path to training data
        --resource_files_test "~/path_to_test_data/test/*.mp4" \    # Path to test data
        --save_model \
        --save_buffer
```

## Tensorboard logs
This will produce 'log' folder, where all the outputs are going to be stored including train/eval logs, tensorboard blobs, and evaluation episode videos. One can attacha tensorboard to monitor training by running:
```
tensorboard --logdir log
```
and opening up tensorboad in your browser.

The console output is also available in a form:
```
| train | E: 1 | S: 1000 | D: 0.8 s | R: 0.0000 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | RLOSS: 0.0000
```
a training entry decodes as:
```
train - training episode
E - total number of episodes 
S - total number of environment steps
D - duration in seconds to train 1 episode
R - episode reward
BR - average reward of sampled batch
ALOSS - average loss of actor
CLOSS - average loss of critic
RLOSS - average reconstruction loss (only if is trained from pixels and decoder)
```
while an evaluation entry:
```
| eval | S: 0 | ER: 21.1676
```
which just tells the expected reward `ER` evaluating current policy after `S` steps. Note that `ER` is average evaluation performance over `num_eval_episodes` episodes (usually 10).
