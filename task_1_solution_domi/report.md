# Introduction
I started with an architecture similar to the original DeepSTARR model. Later, I tested several changes to this baseline. These changes included adding more dropout, using LSTM layers, trying a two-head model for regression and classification and adding more advanced blocks.

# Two Head Architecture
I also experimented with a two-head architecture, where one output head was trained for continuous `rna_dna_ratio` prediction and the second head for `is_active` classification. The idea was to check whether joint learning of regression and classification could improve the results.

In practice, this approach did not outperform the stronger regression-only models. The model tended to make averaged predictions, which decreased results. Because of that, I removed the second head and focused on predicting `rna_dna_ratio` only, while deriving `is_active` from the sign of the prediction.

I also tested a classification-only model, but it also did not outperform the best regression-based models.

During the development process, the codebase went through several versions. Some experimental architectures, such as the two-head model, were removed from the final implementation after they were found to be not effective.

# Data Preparation
The data were loaded from a TSV file. From the original dataset, I kept only the columns `sequence`, `rna_dna_ratio`, and `is_active`. I also removed 15 bases from both ends of each sequence to remove the flanking regions and keep only the central sequence used for modeling.

Each DNA sequence was converted into a one-hot encoded representation. The regression target `rna_dna_ratio` was computed on normalized set using the mean and standard deviation computed only on the training set.

During training, I applied reverse-complement augmentation with probability 0.5 to the training data. I tested this augmentation in the training pipeline and observed a positive effect on model robustness, so I kept it in the final setup. This is biologically justified because a DNA sequence and its reverse complement carry equivalent regulatory information in this setting.

For evaluation, I used stratified splits based on `is_active`, so the class balance was preserved across train, validation, and test sets. 

# Model Development
First, I improved the baseline with additional dropout in the convolutional blocks. Then I tested LSTM-based variants, followed by GELU activations. In the final stage, I moved to stronger convolutional architectures with residual blocks, dilated convolutions, and SE-blocks.

# Summary of Results
The main results are summarized in the table below.

| Type | Model / Fold | Score | Val Loss | MSE | Pearson | Spearman | Acc |
|---|---|---:|---:|---:|---:|---:|---:|
| Single split | OneHeadDeepSTARR | 1.2963 | 0.2888 | 0.4162 | 0.5849 | 0.5784 | 0.7530 |
| Single split | OneHeadDeepSTARRWithAdditionalDropout | 1.3609 | 0.2670 | 0.3851 | 0.6243 | 0.6197 | 0.7750 |
| Single split | OneHeadDeepSTARRWithLSTM | 1.3151 | 0.2901 | 0.4127 | 0.5997 | 0.5926 | 0.7566 |
| Single split | OneHeadDeepSTARRWithLSTMAndAdditionalDropout | 1.3855 | 0.2628 | 0.3734 | 0.6417 | 0.6346 | 0.7812 |
| Single split | OneHeadDeepSTARRWithLSTMAndAdditionalDropoutGELU | 1.3964 | 0.2539 | 0.3645 | 0.6495 | 0.6422 | 0.7834 |
| Single split | OneHeadDeepSTARRWithLSTMAndAdditionalBiggerDropoutGELU | 1.4029 | 0.2547 | 0.3662 | 0.6546 | 0.6477 | 0.7850 |
| Single split | DeepSTARRwithResidualDilatedBlockAndLSTM | 1.4515 | 0.2385 | 0.3288 | 0.6941 | 0.6817 | 0.7902 |
| Single split | DeepSTARRRDSEB | 1.4530 | 0.2376 | 0.3331 | 0.6906 | 0.6833 | 0.7956 |
| 5-fold CV | DeepSTARRRDSEB fold 1 | 1.4340 | 0.2460 | 0.3445 | 0.6757 | 0.6683 | 0.7928 |
| 5-fold CV | DeepSTARRRDSEB fold 2 | 1.4182 | 0.2506 | 0.3523 | 0.6641 | 0.6474 | 0.7894 |
| 5-fold CV | DeepSTARRRDSEB fold 3 | 1.4159 | 0.2375 | 0.3334 | 0.6676 | 0.6597 | 0.7816 |
| 5-fold CV | DeepSTARRRDSEB fold 4 | 1.4034 | 0.2430 | 0.3405 | 0.6632 | 0.6504 | 0.7743 |
| 5-fold CV | DeepSTARRRDSEB fold 5 | 1.4241 | 0.2454 | 0.3453 | 0.6681 | 0.6532 | 0.7906 |

The best single-split model was `DeepSTARRRDSEB`.

| Comparison | Val Loss | MSE | Pearson | Spearman | Acc |
|---|---:|---:|---:|---:|---:|
| DeepSTARRRDSEB vs OneHeadDeepSTARR | -17.7% | -20.0% | +0.106 | +0.105 | +0.043 |

# Cross-Validation and Final Model
After selecting `DeepSTARRRDSEB` as the strongest architecture, I ran 5-fold cross-validation to check whether its performance was stable across different train/validation splits.

The final prediction pipeline uses these five `DeepSTARRRDSEB` fold models as an ensemble. For each test sequence, all five models predict `rna_dna_ratio` and the predictions are averaged. The binary `is_active` label is then derived from the sign of the averaged prediction.

During evaluation and final prediction, I also used reverse-complement augmentation - both the original sequence and its reverse complement were passed through the model, and the two predictions were averaged.

# Results on the Test Set
The final 5-model ensemble achieved the following results on the test set:

| Model | Loss | MSE | Pearson | Spearman | Acc |
|---|---:|---:|---:|---:|---:|
| DeepSTARRRDSEB ensemble | 0.2317 | 0.2954 | 0.7236 | 0.7138 | 0.8117 |

# Dependencies and Evaluation
The project dependencies are listed in `requirements.txt`. They can be installed with:
```bash
pip install -r requirements.txt
```
The final prediction script is `evaluation_script.py`. For the final ensemble model used in this project, the command is:
```bash
python evaluation_script.py models/DeepSTARRRDSEB <path_to_test_data>
```
The model directory must contain the five fold checkpoints:
- `model_fold1_seed42.pt`
- `model_fold2_seed42.pt`
- `model_fold3_seed42.pt`
- `model_fold4_seed42.pt`
- `model_fold5_seed42.pt`

and the normalization statistics file:
- `fold_stats.json`

moreover it requires code from src catalogue.
