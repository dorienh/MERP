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

# no profile

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-2networks/training_noprofile.py \
# --affect_type 'arousals' \
# --num_epochs 30 \
# --model_name 'no_condition' \
# --batch_size 128 \
# --lstm_hidden_dim 512 \
# --lstm_size 10 \
# --step_size 5 \
# --learning_rate 0.001 \
# --conditions 'training'

# with profile - arousal

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'arousals' \
--num_epochs 20 \
--model_name 'condition_age_combined_model_1' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'age'\

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'arousals' \
--num_epochs 20 \
--model_name 'condition_encult_combined_model_1' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'country_enculturation'\

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'arousals' \
--num_epochs 20 \
--model_name 'condition_live_combined_model_1' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'country_live'\

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'arousals' \
--num_epochs 20 \
--model_name 'condition_lang_combined_model_1' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'fav_music_lang'\

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'arousals' \
--num_epochs 20 \
--model_name 'condition_genre_combined_model_1' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'fav_genre'\

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'arousals' \
--num_epochs 20 \
--model_name 'condition_instru_combined_model_1' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'play_instrument'\

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'arousals' \
--num_epochs 20 \
--model_name 'condition_tr_combined_model_1' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'training'\

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'arousals' \
--num_epochs 20 \
--model_name 'condition_trdur_combined_model_1' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'training_duration'\

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'arousals' \
--num_epochs 20 \
--model_name 'condition_trdur_combined_model_1' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'master'\

# with profile - valence

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'valences' \
--num_epochs 20 \
--model_name 'condition_age_combined_model_1' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'age'\

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'valences' \
--num_epochs 20 \
--model_name 'condition_encult_combined_model_1' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'country_enculturation'\

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'valences' \
--num_epochs 20 \
--model_name 'condition_live_combined_model_1' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'country_live'\

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'valences' \
--num_epochs 20 \
--model_name 'condition_lang_combined_model_1' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'fav_music_lang'\

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'valences' \
--num_epochs 20 \
--model_name 'condition_genre_combined_model_1' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'fav_genre'\

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'valences' \
--num_epochs 20 \
--model_name 'condition_instru_combined_model_1' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'play_instrument'\

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'valences' \
--num_epochs 20 \
--model_name 'condition_tr_combined_model_1' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'training'\

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'valences' \
--num_epochs 20 \
--model_name 'condition_trdur_combined_model_1' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'training_duration'\

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'valences' \
--num_epochs 20 \
--model_name 'condition_tr_combined_model_2' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'training'\

/home/meowyan/anaconda3/envs/emo/bin/python \
/home/meowyan/Documents/emotion/method-2networks/training_withprofile.py \
--affect_type 'valences' \
--num_epochs 20 \
--model_name 'condition_trdur_combined_model_1' \
--batch_size 128 \
--lstm_hidden_dim 512 \
--lstm_size 10 \
--step_size 5 \
--learning_rate 0.001 \
--conditions 'master'\