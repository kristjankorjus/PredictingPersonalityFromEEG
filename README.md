Predicting personality from the resting state EEG
=================================================

This repository contains data and code for an article "Personality cannot be predicted from the power of resting state EEG" submitted to a journal Frontiers in Human Neuroscience.

Data is in the folder Data, the code is in the folder Src.

Script `NestedCV.m` does the following:

1. It loads the data
2. It calls a function `Step1_nested_cross_validation.m`, which chooses the best hyper-parameters
3. It predicts personality scores for each personality trait using the function `Step2_cross_validation.m`
4. It performs statistical significance analysis

Note: the full analysis is computationally intensive but it easy very easy to parallelize it.
