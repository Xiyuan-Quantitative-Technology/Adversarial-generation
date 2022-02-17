# Adversarial generations of efficient thermally activated delayed fluorescence molecules
In order to run the code, python packages including (but not limited) deepchem, pytorch, numpy, pandas, and matplotlib should be installed.

In AAE file, adversarial autoencoder is chosen for molecular generation.
In DNN file, the partitioned labeled molecules are used for the DNN training, hyperparametrization, and generalization, respectively.


The result.pkl include AAE model and efficient generative molecules. Following:

with open('result.pkl', 'rb') as f:
    data_list = pickle.load(f)
    f.close()
all_model, all_loss, all_eff_smile, all_smile = data_list
