#!/bin/bash

# method-rdmseg -> lstm without profile

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-rdmseg/training_np_kfold.py \
# --affect_type 'valences' \
# --master 1 \
# --num_epochs 100 \
# --model_name 'semoga_akhir' \
# --batch_size 8 \
# --hidden_dim 512 \
# --num_timesteps 30 \
# --lr 0.0001 \
# --linear true

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-rdmseg/training_np_kfold.py \
# --affect_type 'arousals' \
# --master 1 \
# --num_epochs 100 \
# --model_name 'semoga_akhir' \
# --batch_size 8 \
# --hidden_dim 512 \
# --num_timesteps 30 \
# --lr 0.0001 \
# --linear true


/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-rdmseg/training_np_kfold.py \
--affect_type 'valences' \
--master 0 \
--num_epochs 100 \
--model_name 'bukan_akhir_hd64' \
--batch_size 8 \
--hidden_dim 64 \
--num_timesteps 30 \
--lr 0.0001 \
# --linear true

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-rdmseg/training_np_kfold.py \
--affect_type 'arousals' \
--master 0 \
--num_epochs 100 \
--model_name 'bukan_akhir_hd64' \
--batch_size 8 \
--hidden_dim 64 \
--num_timesteps 30 \
--lr 0.0001 \
# --linear true

# echo "Bash version ${BASH_VERSION}..."
# for i in 'age' 'country_enculturation' 'country_live' 'fav_music_lang' 'gender' 'fav_genre' 'play_instrument' 'training' 'training_duration'
#   do 
#      echo $i
#  done