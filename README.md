# Music Emotion Recognition with Profile Information (MERP)

[![Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/amaai-lab/MERP)
[![Paper](https://img.shields.io/badge/Paper-Sensors%202023-blue)](https://doi.org/10.3390/s23010382)
[![Kaggle](https://img.shields.io/badge/Kaggle-Archive-20BEFF?logo=kaggle)](https://www.kaggle.com/kohenyan/music-emotion-recognition-with-profile-information)

We use MERP to train a baseline emotion prediction model and evaluate the influence of different rater profile features. The repository includes thorough description of the dataset collection process, together with statistics and visualisations.

## Dataset

MERP contains copyright-free full-length musical tracks with dynamic ratings on Russell's two-dimensional valence–arousal model. It was collected through Amazon Mechanical Turk (MTurk). A total of 277 participants were asked to rate 54 selected songs dynamically in terms of valence and arousal, along with their demographic information, listening preferences, and musical background.

50 songs with the most distinctive emotions were selected from the Free Music Archive using a Triple Neural Network with the OpenSmile toolkit, chosen to fully cover the four quadrants of the valence–arousal space. 4 additional songs were selected from DEAM as anchor tracks for quality filtering.

**The cleaned dataset (Parquet + audio) is on [HuggingFace](https://huggingface.co/datasets/amaai-lab/MERP).**  
The original Kaggle dump is archived at [kaggle](https://www.kaggle.com/kohenyan/music-emotion-recognition-with-profile-information).

## Citation

If you find this dataset useful, please cite:

```bibtex
@article{koh2023merp,
  title   = {{MERP}: A Music Dataset with Emotion Ratings and Raters' Profile Information},
  author  = {Koh, Evan Yee and Cheuk, Kin Wai and Heung, Kwan Yee and Agres, Kat R. and Herremans, Dorien},
  journal = {Sensors},
  volume  = {23},
  number  = {1},
  pages   = {382},
  year    = {2023},
  doi     = {10.3390/s23010382}
}
```

## Folder structure

```
MERP
├── analysis/codes
│     ├── song_selection.py
│     └── va_result_plotting.py
├── method-hilang
│     ├── dataloader.py
│     ├── network.py
│     └── training.py
├── method-lstm
│     ├── dataloader.py
│     ├── network.py
│     └── training_np.py
├── processing
│     ├── ave_exp_by_prof.py
│     ├── extract_audio_feats.py
│     └── extract_exps.py
├── amazon_Merged.html
└── util.py
```

- `analysis/codes` — analysis and visualisation scripts (figures used in the paper)
- `method-hilang` — dataloader, model architecture and training script for the fully connected model
- `method-lstm` — dataloader, model architecture and training script for the LSTM model
- `processing` — data processing: feature extraction, label averaging across raters
- `amazon_Merged.html` — HTML template for the MTurk listening study
- `util.py` — bin values for categorical profile features
