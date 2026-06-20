# Music Emotion Recognition with Profile Information (MERP)

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-amaai--lab/MERP-yellow)](https://huggingface.co/datasets/amaai-lab/MERP)
[![Paper](https://img.shields.io/badge/Paper-Sensors%202023-blue)](https://doi.org/10.3390/s23010382)

> Music is capable of conveying many emotions. The level and type of emotion perceived by a listener, however, is highly subjective. In this study, we present the **Music Emotion Recognition with Profile information dataset (MERP)**. This database was collected through Amazon Mechanical Turk (MTurk) and features dynamical valence and arousal ratings of 54 selected full-length songs. The songs were selected from the Free Music Archive using an innovative method — a Triple Neural Network with the OpenSmile toolkit — to identify 50 songs with the most distinctive emotions, chosen to fully cover the four quadrants of the valence–arousal space. Four additional songs were selected from DEAM as anchor tracks for quality filtering. A total of 452 participants annotated the dataset, with 277 remaining after thorough cleaning. Their demographic information, listening preferences, and musical background were recorded. We offer an extensive analysis of the dataset together with baseline emotion prediction models.

---

## Dataset

**The cleaned dataset (Parquet + audio) is on [HuggingFace](https://huggingface.co/datasets/amaai-lab/MERP).**

| Property | Value |
| --- | --- |
| Songs | 54 full-length tracks (50 × FMA + 4 × DEAM) |
| Raters | 277 (after cleaning); 452 total |
| Annotation | Continuous valence & arousal at 10 Hz, scale [−1, 1] |
| Profile fields | Age, gender, country, language, genre, instrument, music training |

---

## Models

The paper evaluates several architectures for predicting continuous valence/arousal from audio features (OpenSmile ComParE) — with and without rater profile information concatenated to the feature vector at each timestep.

### Main models (paper results)

| Folder | Architecture | Profile | Evaluation |
| --- | --- | --- | --- |
| `method-rdmseg/` | 2-layer BiLSTM + FC | No | k-fold, random segment sampling |
| `method-rdmseg-prof/` | 2-layer BiLSTM + FC | Yes | k-fold, random segment sampling |

The random-segment approach samples fixed-length segments from each song in every epoch, ensuring full-song coverage without padding.

### Baselines

| Folder | Architecture | Notes |
| --- | --- | --- |
| `method/` | 2-layer FC | Averaged labels per song; train/test split |
| `method-lstm/` | Single LSTM | Windowed input; no profile |
| `method-hilang/` | 3-layer FC | Random segments; early prototype |

### Additional experiments

| Folder | Architecture | Notes |
| --- | --- | --- |
| `method-2networks/` | Single LSTM | Non-averaged labels; with/without profile |
| `method-time/` | Single LSTM | Time-feature variant |
| `method-10fold/` | — | Placeholder / stub |

---

## Repository structure

```
MERP-src/
├── processing/
│   ├── extract_exps.py          # parse raw MTurk CSVs → annotation pickle
│   ├── extract_pinfo.py         # parse rater profile info
│   ├── pruning.py               # 7-step quality filtering + rescaling + smoothing
│   ├── extract_audio_feats.py   # OpenSmile feature extraction
│   └── ave_exp_by_prof.py       # average annotations by rater subgroup
│
├── analysis/codes/
│   ├── song_selection.py        # Triple Neural Network for song selection
│   ├── krippendorff.py          # inter-rater reliability
│   ├── anova.py / kruskal_wallis.py  # statistical tests by profile group
│   ├── deam_comparison.py       # benchmark against DEAM annotations
│   └── va_result_plotting.py    # valence–arousal visualisations
│
├── method-rdmseg-prof/          # ← main paper model (BiLSTM + profile)
├── method-rdmseg/               # ← main paper model (BiLSTM, no profile)
├── method/                      # FC baseline
├── method-lstm/                 # LSTM baseline
├── method-hilang/               # FC prototype
├── method-2networks/            # 2-network experiment
├── method-time/                 # time-feature experiment
│
├── amazon_Merged.html           # MTurk HIT template
├── util.py                      # category bin values for profile features
└── util_method.py               # shared training utilities
```

---

## Citation

```bibtex
@article{koh2023merp,
  title   = {{MERP}: A Music Dataset with Emotion Ratings and Raters' Profile Information},
  author  = {Koh, En Yan and Cheuk, Kin Wai and Heung, Kwan Yee and Agres, Kat R. and Herremans, Dorien},
  journal = {Sensors},
  volume  = {23},
  number  = {1},
  pages   = {382},
  year    = {2023},
  doi     = {10.3390/s23010382}
}
```
