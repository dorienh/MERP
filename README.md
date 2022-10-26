# Music Emotion Recognition with Profile Information (MERP)

Welcome to the repository for MERP. 

We use MERP to train a baseline emotion prediction model and evaluate the influence of the different profile features. We provide a thorough description of the dataset collection process, together with statistics and visualisations

## Citation

If you find this dataset useful, please cite our [original paper](https://www.preprints.org/manuscript/202210.0301/v1): 

Koh, E.; Cheuk, K.W.; Heung, K.Y.; Agres, K.R.; Herremans, D. MERP: A Dataset of Emotion Ratings for Full-Length Musical Songs with Profile Information of Raters. Preprints 2022, 2022100301 (doi: 10.20944/preprints202210.0301.v1).



## Dataset

MERP contains copyright-free full-length musical tracks with dynamic ratings on Russell's two-dimensional valence and arousal mode. 
It was collected through Amazon Mechanical Turk (MTurk). A total of 277 participants were asked to rate 54 selected songs dynamically in terms of valence and arousal. This dataset contains music features, as well as profile information of the annotators (their demographic information, listening preferences, and musical background were recorded).

50 songs with the most distinctive emotions were selected from the Free Music Archive by using a Triple Neural Network with the OpenSmile toolkit. Specifically, the songs were chosen to fully cover the four quadrants of the valence arousal space. 4 additional songs were selected from DEAM to act as a benchmark in order to filter out low quality ratings. 

You can access MERP via [kaggle](https://www.kaggle.com/kohenyan/music-emotion-recognition-with-profile-information)

## Folder structure 

```
MERP
├──analysis/codes
│     ├─song_selection.py
│     ├─va_result_plotting.py
│     │
│
├──method-hilang
│     ├─dataloader.py
│     ├─network.py
│     ├─training.py
│     │
│
├──method-lstm
│     ├─dataloader.py
│     ├─network.py
│     ├─training_np.py
│     │
│
├──processing
│     ├─ave_exp_by_prof.py
│     ├─extract_audio_feats.py
│     ├─extract_exps.py
│     │
│
├──amazon_Merged.html
├──util.py
│   
```
- `analysis/codes`: python file for analysis and visualization (e.g. figures that used in the article)

- `method-hilang`: dataloader, model architecture and training script for the fully connected model

- `method-lstm`: dataloader, model architecture and training script for the long short-term memory 

- `processing`: python files for data processing which include extracting music features from the songs, label averaging for all raters per song

- `amazon_Merged.html`: html code of the listening study with MTurk

- `util.py`: this file includes the bin value of categories in each profile features 




