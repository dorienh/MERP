#!/bin/bash

# 'age', 'country_enculturation', 'country_live', 'fav_music_lang', 'gender', 'fav_genre', 'play_instrument', 'training', 'training_duration'


# no profile

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_noprofile.py \
# --affect_type 'valences' \
# --num_epochs 30 \
# --model_name 'no_condition_mean' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --mean true \
# --median ''\

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_noprofile.py \
# --affect_type 'valences' \
# --num_epochs 30 \
# --model_name 'no_condition_median' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --mean '' \
# --median true \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_noprofile.py \
# --affect_type 'valences' \
# --num_epochs 30 \
# --model_name 'no_condition_no_agg' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --mean '' \
# --median '' \


# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_noprofile.py \
# --affect_type 'valences' \
# --num_epochs 30 \
# --model_name 'no_condition_mean_lr_005' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.005 \
# --mean true \
# --median '' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_noprofile.py \
# --affect_type 'valences' \
# --num_epochs 30 \
# --model_name 'no_condition_median_lr_005' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.005 \
# --mean '' \
# --median true \

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_noprofile.py \
--affect_type 'valences' \
--num_epochs 1000 \
--model_name 'no_condition_no_agg_lr_005_mean_1000epochs' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.005 \
--mean true \
--median '' \

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_noprofile.py \
--affect_type 'valences' \
--num_epochs 1000 \
--model_name 'no_condition_no_agg_lr_005_1000epochs' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.005 \
--mean '' \
--median '' \

# with profile - arousal

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'valences' \
--num_epochs 1000 \
--model_name 'instru_tr_trdur_model_2_lr_005_mean_1000epoch' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.005 \
--conditions 'play_instrument' 'training' 'training_duration' \
--mean true \
--median '' \

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'valences' \
--num_epochs 1000 \
--model_name 'instru_tr_trdur_model_2_lr_005_1000epoch' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.005 \
--conditions 'play_instrument' 'training' 'training_duration' \
--mean '' \
--median '' \


# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'encult_model_2_mean' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'country_enculturation' \
# --mean true \
# --median '' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'live_model_2_mean' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'country_live' \
# --mean true \
# --median '' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'lang_model_2_mean' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'fav_music_lang' \
# --mean true \
# --median '' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'gender_model_2_mean' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'gender' \
# --mean true \
# --median '' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'genre_model_2_mean' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'fav_genre' \
# --mean true \
# --median '' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'instru_model_2_mean' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'play_instrument' \
# --mean true \
# --median '' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'tr_model_2_mean' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'training' \
# --mean true \
# --median '' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'trdur_model_2_mean' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'training_duration' \
# --mean true \
# --median '' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'master_model_2_mean' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'master' \
# --mean true \
# --median '' \

# # with profile - valence

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_age_combined_model_2' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'age'\

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_encult_combined_model_2' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'country_enculturation'\

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_live_combined_model_2' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'country_live'\

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_lang_combined_model_2' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'fav_music_lang'\

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_genre_combined_model_2' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'fav_genre'\

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_instru_combined_model_2' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'play_instrument'\

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_tr_combined_model_2' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'training'\

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_trdur_combined_model_2' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'training_duration'\

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_tr_combined_model_2' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'training'\

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_trdur_combined_model_2' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'master'\