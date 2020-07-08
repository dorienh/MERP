#!/bin/bash

# 'age', 'country_enculturation', 'country_live', 'fav_music_lang', 'gender', 'fav_genre', 'play_instrument', 'training', 'training_duration'




# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
# --affect_type 'arousals' \
# --num_epochs 10 \
# --model_name 'condition_tr_combined_model_1' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'training'\


/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'arousals' \
--num_epochs 10 \
--model_name 'condition_tr_combined_model_2' \
--batch_size 32 \
--hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'training'\