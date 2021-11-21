#!/bin/bash
for i in 'age' 'gender' 'country_live' 'country_enculturation' 'fav_music_lang' 'fav_genre' 'play_instrument' 'training' 'training_duration' 'master'
# for i in 'age' 'gender' 'residence' 'enculturation' 'language' 'genre' 'instrument' 'training' 'duration' 'master'
do
    /home/meowyan/anaconda3/envs/emo/bin/python \
    /home/meowyan/Documents/emotion/method-rdmseg-prof/testing_kfold.py \
    --affect_type 'valences' \
    --linear 'True' \
    --model_name "hd512_mse1_smooth15_$i" \
    --hidden_dim 512 \
    --conditions "$i"
done
# for i in 'age' 'gender' 'country_live' 'country_enculturation' 'fav_music_lang' 'fav_genre' 'play_instrument' 'training' 'training_duration' 'master'
# do 
#     /home/meowyan/anaconda3/envs/emo/bin/python \
#     /home/meowyan/Documents/emotion/method-rdmseg-prof/testing_kfold.py \
#     --affect_type 'valences' \
#     --model_name "hd512_mse1_smooth15_$i" \
#     --hidden_dim 512 \
#     --conditions "$i" \
# done