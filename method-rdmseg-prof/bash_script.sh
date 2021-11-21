#!/bin/bash

# method-rdmseg-prof -> linear and lstm with profile

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-rdmseg-prof/training_linear_kfold.py \
# --affect_type 'valences' \
# --num_epochs 100 \
# --model_name "5fold_all_profiles" \
# --batch_size 8 \
# --hidden_dim 512 \
# --num_timesteps 30 \
# --learning_rate 0.0001 \
# --conditions "age" "gender" "country_live" "country_enculturation" "fav_music_lang" "fav_genre" "play_instrument" "training" "training_duration" "master" \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-rdmseg-prof/training_linear_kfold.py \
# --affect_type 'arousals' \
# --num_epochs 100 \
# --model_name "5fold_all_profiles" \
# --batch_size 8 \
# --hidden_dim 512 \
# --num_timesteps 30 \
# --learning_rate 0.0001 \
# --conditions "age" "gender" "country_live" "country_enculturation" "fav_music_lang" "fav_genre" "play_instrument" "training" "training_duration" "master" \

# "age" "gender" "country_live" "country_enculturation" "fav_music_lang" "fav_genre" "play_instrument" "training" "training_duration" "master"

# echo "Bash version ${BASH_VERSION}..."


for i in 'age' 'gender' 'residence' 'enculturation' 'language' 'genre' 'instrument' 'training' 'duration' 'master'
#   do 
# for i in 'training' 'duration' 'master'
  do
    /home/meowyan/anaconda3/envs/emo/bin/python \
    /home/meowyan/Documents/emotion/method-rdmseg-prof/training_kfold.py \
    --affect_type 'valences' \
    --master 0 \
    --num_epochs 100 \
    --model_name "hd128_$i" \
    --batch_size 8 \
    --hidden_dim 128 \
    --num_timesteps 30 \
    --lr 0.0001 \
    --conditions "$i" \
    --plot true \
    # --linear \


    /home/meowyan/anaconda3/envs/emo/bin/python \
    /home/meowyan/Documents/emotion/method-rdmseg-prof/training_kfold.py \
    --affect_type 'arousals' \
    --master 0 \
    --num_epochs 100 \
    --model_name "hd128_$i" \
    --batch_size 8 \
    --hidden_dim 128 \
    --num_timesteps 30 \
    --lr 0.0001 \
    --conditions "$i" \
    --plot true \
    # --linear \

    # /home/meowyan/anaconda3/envs/emo/bin/python \
    # /home/meowyan/Documents/emotion/method-rdmseg-prof/training_kfold.py \
    # --affect_type 'valences' \
    # --master 1 \
    # --num_epochs 100 \
    # --model_name "hd512_mse_smooth15_$i" \
    # --batch_size 8 \
    # --hidden_dim 512 \
    # --num_timesteps 30 \
    # --lr 0.0001 \
    # --conditions "$i" \
    # --plot true \
    # --linear true\


    # /home/meowyan/anaconda3/envs/emo/bin/python \
    # /home/meowyan/Documents/emotion/method-rdmseg-prof/training_kfold.py \
    # --affect_type 'arousals' \
    # --master 1 \
    # --num_epochs 100 \
    # --model_name "hd512_mse_smooth15_$i" \
    # --batch_size 8 \
    # --hidden_dim 512 \
    # --num_timesteps 30 \
    # --lr 0.0001 \
    # --conditions "$i" \
    # --plot true \
    # --linear true\

    # /home/meowyan/anaconda3/envs/emo/bin/python \
    # /home/meowyan/Documents/emotion/method-rdmseg-prof/training_2lstm_kfold.py \
    # --affect_type 'valences' \
    # --num_epochs 100 \
    # --model_name "5fold_bidir_hd512_mse_$i" \
    # --batch_size 8 \
    # --hidden_dim 512 \
    # --lstm_size 30 \
    # --learning_rate 0.0001 \
    # --conditions "$i" \


    # /home/meowyan/anaconda3/envs/emo/bin/python \
    # /home/meowyan/Documents/emotion/method-rdmseg-prof/training_2lstm_kfold.py \
    # --affect_type 'arousals' \
    # --num_epochs 100 \
    # --model_name "5fold_bidir_hd512_mse_$i" \
    # --batch_size 8 \
    # --hidden_dim 512 \
    # --lstm_size 30 \
    # --learning_rate 0.0001 \
    # --conditions "$i" \

    #  echo $i
done

date +"%Y-%m-%d %T"