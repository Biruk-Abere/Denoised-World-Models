#!/bin/bash

# Weight coefficients for the four losses
w_rec="1e-2"
w_ub="1e-3"
w_enc="1e-1"
w_lb="1e-2"

# Array of seeds for randomness in training
declare -a seeds=("0" "1" "2")

# Loop through each seed to run the training script
for seed in "${seeds[@]}"
do
    # Execute the Python training script with the following parameters:
    python train.py \
        --domain_name cheetah \                                     # Domain name for the training
        --task_name run \                                           # Task name
        --encoder_type pixel \                                      # Type of encoder to use
        --decoder_type pixel \                                      # Type of decoder to use
        --action_repeat 4 \                                         # How many times to repeat the action
        --save_video \                                              # Save training as video
        --save_tb \                                                 # Save TensorBoard logs
        --work_dir ./log \                                          # Directory to save logs and outputs
        --seed $seed \                                              # Seed for randomness
        --w_enc $w_enc \                                            # Weight for I_LTC
        --w_rec $w_rec \                                            # Weight for I_Rec
        --w_ub $w_ub \                                              # Weight for I_CLUB
        --w_lb $w_lb \                                              # Weight for I_NCEe
        --img_source video \                                        # Source for images
        --resource_files "~/path_to_training_data/train/*.mp4" \    # Path to training data
        --resource_files_test "~/path_to_test_data/test/*.mp4" \    # Path to test data
        --save_model \                                              # Save the trained model
        --save_buffer                                               # Save the replay buffer
done
