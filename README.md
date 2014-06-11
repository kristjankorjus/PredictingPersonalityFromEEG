Predicting personality from the resting state EEG
=================================================

This repository contains data and code for an article "Predicting personality from the resting state EEG" submitted to a journal Frontiers in Human Neuroscience.

Everything is in the folder "Scripts, functions, data and results".

1.  Script `Step1_crossvalidation.m` loads data and runs cross-validation with lots of different hyper-parameters. Running the script takes about 20 hours. It saves files `Partitions.m` and `Results.m`. Current files in the folder are the same which were used in the paper.
2.  Next script `Step2_crosstesting.m` can be use to test the best combination of hyper-parameters for the rest of the data. It saves the results to a file called `Errors.mat`.


