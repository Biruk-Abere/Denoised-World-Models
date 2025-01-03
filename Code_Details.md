## Code Details for the Reviewers
### Natural Backgrounds: 
[dmc2gym/natural_imgsource.py](dmc2gym/natural_imgsource.py):\
This file provides the class that is used to generate backgrounds for the natural background setting

### Removing Ground Plane:
In order to remove the ground plane, we add `rgba="0 0 0 0"` to all xml files in `local_dm_control_suite` folder.
```xml
<geom name="ground" type="plane" conaffinity="1" pos="98 0 0" size="100 .8 .5" material="grid" rgba="0 0 0 0"/>
```

### Models:
[Encoder Model](encoder.py): Line 13

[Transition Model](encoder.py): Line 137


### Losses
[NCE Loss](dpi.py): Line 185

[CLUB Loss](encoder.py): Line 216

## For running Natural Settings
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
## For running Random Settings
For random settings, the random_bg variable in line 126 of [dmc2gym/wrappers.py](dmc2gym/wrappers.py) should be set to True.

## Tensorboard logs

This will produce 'log' folder, where all the outputs are going to be stored including train/eval logs, tensorboard blobs, and evaluation episode videos. One can attacha tensorboard to monitor training by running:
```
tensorboard --logdir log
```
and opening up tensorboad in your browser.