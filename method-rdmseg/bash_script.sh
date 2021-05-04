#!/bin/bash

# method-rdmseg -> lstm without profile

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-rdmseg/training.py \
--affect_type 'valences' \
--num_epochs 1000 \
--model_name '2fc_mse10_lagi' \
--batch_size 8 \
--hidden_dim 512 \
--lstm_size 10 \
--mse_weight 10.0 \
--learning_rate 0.0001 \


/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-rdmseg/training.py \
--affect_type 'arousals' \
--num_epochs 1000 \
--model_name '2fc_mse10_lagi' \
--batch_size 8 \
--hidden_dim 512 \
--lstm_size 10 \
--mse_weight 10.0 \
--learning_rate 0.0001 \

# echo "Bash version ${BASH_VERSION}..."
# for i in 'age' 'country_enculturation' 'country_live' 'fav_music_lang' 'gender' 'fav_genre' 'play_instrument' 'training' 'training_duration'
#   do 
#      echo $i
#  done