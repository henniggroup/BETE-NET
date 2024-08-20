# BETE-NET
This repo contains the models, training data and test predictions for the work described in the paper: [Accelerating superconductor discovery through tempered deep learning of the electron-phonon spectral function](https://arxiv.org/abs/2401.16611)

## Files and directories

The `CSO/` `CPD/` and `FPD/` directories contain the trained models described in the paper.

The `database.json` file contains the database used to train the model.

The `structures` directory contains the corresponding cif files.

The `indices/` directory contain the training `indices/idx_train_full.txt`, and testing `indices/idx_test_full.txt` indices as well as the indices used for bootstrapping.

The `notebook/` directory contains notebooks to visualize the predictions, train the models and make predictions on a dataset.

The `test_preds` directory contain the predictions published in the paper.
