#!/bin/bash

# 'age', 'country_enculturation', 'country_live', 'fav_music_lang', 'gender', 'fav_genre', 'play_instrument', 'training', 'training_duration'


# epoch variation

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'arousals' \
# --num_epochs 10 \
# --model_name 'epochs_10' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'age' 'gender' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'arousals' \
# --num_epochs 50 \
# --model_name 'epochs_50' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'age' 'gender' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'arousals' \
# --num_epochs 100 \
# --model_name 'epochs_100' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'age' 'gender' \

# batch_size variation

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'arousals' \
# --num_epochs 20 \
# --model_name 'batchsize_4' \
# --batch_size 4 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'age' 'gender' \

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-time/training_withprofile.py \
--affect_type 'arousals' \
--num_epochs 20 \
--model_name 'batchsize_16' \
--batch_size 16 \
--hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-time/training_withprofile.py \
--affect_type 'arousals' \
--num_epochs 20 \
--model_name 'batchsize_32' \
--batch_size 32 \
--hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-time/training_withprofile.py \
--affect_type 'arousals' \
--num_epochs 20 \
--model_name 'batchsize_64' \
--batch_size 64 \
--hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-time/training_withprofile.py \
--affect_type 'arousals' \
--num_epochs 20 \
--model_name 'batchsize_128' \
--batch_size 128 \
--hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \

# conditions variation

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-time/training_withprofile.py \
--affect_type 'arousals' \
--num_epochs 20 \
--model_name 'condition_age' \
--batch_size 32 \
--hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'age' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'arousals' \
# --num_epochs 20 \
# --model_name 'condition_gender' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'gender' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'arousals' \
# --num_epochs 20 \
# --model_name 'condition_encul' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'country_enculturation' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'arousals' \
# --num_epochs 20 \
# --model_name 'condition_live' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'country_live' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'arousals' \
# --num_epochs 20 \
# --model_name 'condition_genre' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'fav_genre' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'arousals' \
# --num_epochs 20 \
# --model_name 'condition_play' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'play_instrument' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'arousals' \
# --num_epochs 20 \
# --model_name 'condition_tr' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'training' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'arousals' \
# --num_epochs 20 \
# --model_name 'condition_trdur' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'training_duration' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'arousals' \
# --num_epochs 20 \
# --model_name 'condition_all' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'age' 'country_enculturation' 'country_live' 'fav_music_lang' 'gender' 'fav_genre' 'play_instrument' 'training' 'training_duration' \


# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'arousals' \
# --num_epochs 20 \
# --model_name 'condition_trdur_tr' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'training' 'training_duration' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'arousals' \
# --num_epochs 20 \
# --model_name 'condition_trdur_tr' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'country_live' 'country_enculturation' \


# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'arousals' \
# --num_epochs 20 \
# --model_name 'condition_trdur_tr' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'play_instrument' 'fav_genre' \


# # valences

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'batchsize_16' \
# --batch_size 16 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'batchsize_32' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'batchsize_64' \
# --batch_size 64 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'batchsize_128' \
# --batch_size 128 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \

# # conditions variation

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_age' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'age' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_gender' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'gender' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_encul' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'country_enculturation' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_live' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'country_live' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_genre' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'fav_genre' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_play' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'play_instrument' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_tr' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'training' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_trdur' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'training_duration' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_all' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'age' 'country_enculturation' 'country_live' 'fav_music_lang' 'gender' 'fav_genre' 'play_instrument' 'training' 'training_duration' \


# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_trdur_tr' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'training' 'training_duration' \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_trdur_tr' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'country_live' 'country_enculturation' \


# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-time/training_withprofile.py \
# --affect_type 'valences' \
# --num_epochs 20 \
# --model_name 'condition_trdur_tr' \
# --batch_size 32 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'play_instrument' 'fav_genre' \
