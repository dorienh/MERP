#!/bin/bash

# method-rdmseg-prof -> linear and lstm with profile

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-rdmseg-prof/training_lstm_2fc.py \
# --affect_type 'valences' \
# --num_epochs 1000 \
# --model_name "lstm2fc_mse10_exp" \
# --batch_size 8 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --learning_rate 0.0001 \
# --mse_weight 10.0 \
# --conditions "play_instrument" "training" "training_duration" \

# /home/meowyan/anaconda3/envs/emo/bin/python \
# /home/meowyan/Documents/emotion/method-rdmseg-prof/training_linear.py \
# --affect_type 'valences' \
# --num_epochs 1000 \
# --model_name "mse10_exp" \
# --batch_size 8 \
# --hidden_dim 512 \
# --lstm_size 10 \
# --learning_rate 0.0001 \
# --mse_weight 10.0 \
# --conditions "play_instrument" "training" "training_duration" \

# echo "Bash version ${BASH_VERSION}..."
for i in 'age' 'gender' 'country_live' 'country_enculturation' 'fav_music_lang' 'fav_genre' 'play_instrument' 'training' 'training_duration' 'master'
  do 

    # /home/meowyan/anaconda3/envs/emo/bin/python \
    # /home/meowyan/Documents/emotion/method-rdmseg-prof/training_linear_kfold.py \
    # --affect_type 'valences' \
    # --num_epochs 100 \
    # --model_name "hd512_mse1_smooth15_$i" \
    # --batch_size 8 \
    # --hidden_dim 512 \
    # --num_timesteps 30 \
    # --learning_rate 0.0001 \
    # --conditions "$i" \


    # /home/meowyan/anaconda3/envs/emo/bin/python \
    # /home/meowyan/Documents/emotion/method-rdmseg-prof/training_linear_kfold.py \
    # --affect_type 'arousals' \
    # --num_epochs 100 \
    # --model_name "hd512_mse1_smooth15_$i" \
    # --batch_size 8 \
    # --hidden_dim 512 \
    # --num_timesteps 30 \
    # --learning_rate 0.0001 \
    # --conditions "$i" \

    /home/meowyan/anaconda3/envs/emo/bin/python \
    /home/meowyan/Documents/emotion/method-rdmseg-prof/training_2lstm_kfold.py \
    --affect_type 'valences' \
    --num_epochs 100 \
    --model_name "5fold_bidir_hd512_mse_$i" \
    --batch_size 8 \
    --hidden_dim 512 \
    --lstm_size 30 \
    --learning_rate 0.0001 \
    --conditions "$i" \


    /home/meowyan/anaconda3/envs/emo/bin/python \
    /home/meowyan/Documents/emotion/method-rdmseg-prof/training_2lstm_kfold.py \
    --affect_type 'arousals' \
    --num_epochs 100 \
    --model_name "5fold_bidir_hd512_mse_$i" \
    --batch_size 8 \
    --hidden_dim 512 \
    --lstm_size 30 \
    --learning_rate 0.0001 \
    --conditions "$i" \

    # /home/meowyan/anaconda3/envs/emo/bin/python \
    # /home/meowyan/Documents/emotion/method-rdmseg-prof/training_linear.py \
    # --affect_type 'valences' \
    # --num_epochs 1000 \
    # --model_name "mse1_$i" \
    # --batch_size 8 \
    # --hidden_dim 512 \
    # --lstm_size 1 \
    # --learning_rate 0.0001 \
    # --mse_weight 1.0 \
    # --conditions "$i" \


    # /home/meowyan/anaconda3/envs/emo/bin/python \
    # /home/meowyan/Documents/emotion/method-rdmseg-prof/training_linear.py \
    # --affect_type 'arousals' \
    # --num_epochs 1000 \
    # --model_name "mse1_$i" \
    # --batch_size 8 \
    # --hidden_dim 512 \
    # --lstm_size 1 \
    # --learning_rate 0.0001 \
    # --mse_weight 1.0 \
    # --conditions "$i" \

    # /home/meowyan/anaconda3/envs/emo/bin/python \
    # /home/meowyan/Documents/emotion/method-rdmseg-prof/training_lstm.py \
    # --affect_type 'valences' \
    # --num_epochs 1000 \
    # --model_name "lstm1fc_mse1_$i" \
    # --batch_size 8 \
    # --hidden_dim 512 \
    # --lstm_size 10 \
    # --learning_rate 0.0001 \
    # --mse_weight 1.0 \
    # --conditions "$i" \


    # /home/meowyan/anaconda3/envs/emo/bin/python \
    # /home/meowyan/Documents/emotion/method-rdmseg-prof/training_lstm.py \
    # --affect_type 'arousals' \
    # --num_epochs 1000 \
    # --model_name "lstm1fc_mse1_$i" \
    # --batch_size 8 \
    # --hidden_dim 512 \
    # --lstm_size 10 \
    # --learning_rate 0.0001 \
    # --mse_weight 1.0 \
    # --conditions "$i" \

    # /home/meowyan/anaconda3/envs/emo/bin/python \
    # /home/meowyan/Documents/emotion/method-rdmseg-prof/training_lstm_2fc.py \
    # --affect_type 'valences' \
    # --num_epochs 1000 \
    # --model_name "lstm2fc_mse1_$i" \
    # --batch_size 8 \
    # --hidden_dim 512 \
    # --lstm_size 10 \
    # --learning_rate 0.0001 \
    # --mse_weight 1.0 \
    # --conditions "$i" \


    # /home/meowyan/anaconda3/envs/emo/bin/python \
    # /home/meowyan/Documents/emotion/method-rdmseg-prof/training_lstm_2fc.py \
    # --affect_type 'arousals' \
    # --num_epochs 1000 \
    # --model_name "lstm2fc_mse1_$i" \
    # --batch_size 8 \
    # --hidden_dim 512 \
    # --lstm_size 10 \
    # --learning_rate 0.0001 \
    # --mse_weight 1.0 \
    # --conditions "$i" \

    
    # #############
    
    
    
    # /home/meowyan/anaconda3/envs/emo/bin/python \
    # /home/meowyan/Documents/emotion/method-rdmseg-prof/training_lstm.py \
    # --affect_type 'valences' \
    # --num_epochs 1000 \
    # --model_name "lstm1fc_mse10_$i" \
    # --batch_size 8 \
    # --hidden_dim 512 \
    # --lstm_size 10 \
    # --learning_rate 0.0001 \
    # --mse_weight 10.0 \
    # --conditions "$i" \


    # /home/meowyan/anaconda3/envs/emo/bin/python \
    # /home/meowyan/Documents/emotion/method-rdmseg-prof/training_lstm.py \
    # --affect_type 'arousals' \
    # --num_epochs 1000 \
    # --model_name "lstm1fc_mse10_$i" \
    # --batch_size 8 \
    # --hidden_dim 512 \
    # --lstm_size 10 \
    # --learning_rate 0.0001 \
    # --mse_weight 10.0 \
    # --conditions "$i" \

    # /home/meowyan/anaconda3/envs/emo/bin/python \
    # /home/meowyan/Documents/emotion/method-rdmseg-prof/training_lstm_2fc.py \
    # --affect_type 'valences' \
    # --num_epochs 1000 \
    # --model_name "lstm2fc_mse10_$i" \
    # --batch_size 8 \
    # --hidden_dim 512 \
    # --lstm_size 10 \
    # --learning_rate 0.0001 \
    # --mse_weight 10.0 \
    # --conditions "$i" \


    # /home/meowyan/anaconda3/envs/emo/bin/python \
    # /home/meowyan/Documents/emotion/method-rdmseg-prof/training_lstm_2fc.py \
    # --affect_type 'arousals' \
    # --num_epochs 1000 \
    # --model_name "lstm2fc_mse10_$i" \
    # --batch_size 8 \
    # --hidden_dim 512 \
    # --lstm_size 10 \
    # --learning_rate 0.0001 \
    # --mse_weight 10.0 \
    # --conditions "$i" \

    #  echo $i
done

