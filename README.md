# summarize_cluster_overlap

Python programs for merging mixture components and summarizing their results.
Currently, it targets the problem of the Gaussian mixtures.
Programs for running the comparison methods are also provided.

## Required packages

Required python packages are listed in `Pipfile` and `Pipfile.lock` files.
Note that additional installments are needed for loading real datasets
and running comparison methods.
Refer to the section Real dataset and Comparison Method for them.

## Usage

### Overview
* The function `load_data` is used to load and process the real dataset.
* The classes `GMMUtils` and `GMMModelSelection` are used to create the probabilities of latent variables from the predefined and estimated Gaussian mixture models, respectively.
* The classes `(Ent / Nent1 / DEMP / DEMP2 / MC / NMC)MergeComponents`
handle the merging algorithm
with their method `fit`.
Their method `clustering_summarization` create the clustering summary of the fitted data.

### Jupyter

We provide the following five `.ipynb` files to show the usage of this program for experiments. All of the experimental results can be reproduced by executing them.

* `experiment_artificial.ipynb` ... Experiment for the artificial dataset.
* `experiment_real.ipynb` ... Comparing F-measure and ARI for the real dataset.
* `dataset_outlier.ipynb` ... Investigating the relationships between the real datasets and proportion of outlier points.
* `clustering_summarization.ipynb` ... Showing how to obtain the clustering summarization.
* `correlation_mc_nmc_entropy.ipynb` ... Investigating the correlations between MC/NMC and entropy.

## Real dataset

We use eight open datasets in the real data experiment
and the function `load_data` loads and pre-processes the datasets.
To use this function, it is need to download data files or install packages
so that it can obtain the dataset.
We list the source and required procedures for each dataset below.

* The **AIS** dataset is loaded from the R package
`locfit` (https://cran.r-project.org/web/packages/locfit/index.html).
You need to install R with this package on your machine.
* The **flea beetles** dataset is loaded from the R package
`fdm2id` (https://cran.r-project.org/web/packages/fdm2id/index.html).
You need to install R with this package on your machine.
* The **crabs** dataset is loaded from the R package
`MASS` (https://cran.r-project.org/web/packages/MASS/index.html).
You need to install R with this package on your machine.
* The **DLCBL** dataset is loaded from the R package
`EMMIXuskew` (https://cran.r-project.org/web/packages/EMMIXuskew/index.html).
You need to install R with this package on your machine.
* The **ecoli** dataset is downloaded from
https://archive.ics.uci.edu/ml/datasets/ecoli.
You need to download `ecoli.data` from there
and put it in the `datasets` directory.
* The **seeds** dataset is downloaded from
https://archive.ics.uci.edu/ml/datasets/seeds.
You need to download `seeds_dataset.txt` from there
and put it in the `datasets` directory.
* The **yeast** dataset is downloaded from
http://archive.ics.uci.edu/ml/datasets/yeast.
You need to download `yeast.data` from there
and put it in the `datasets` directory.
* The **wisconsin breast cancer** dataset is loaded
from the function the in the `scikit-learn` package.
Additional procedures are not required.

# Comparison Method

`GMMModelSelection` can be used for GMM + BIC and GMM + DNML
by setting `mode='GMM_BIC'` and `mode='GMM_DNML'`, respectively.
`MixtureOfMixture` is a wrapper of MixMix. To run this, additional R packages are required. See [1] for the datail.

# References

[1] Gertraud Malsiner-Walli, Sylvia Fruhwirth-Schnatter and Bettina Grun.
Identifying Mixtures of Mixtures Using Bayesian Estimation.
*Journal of Computational and Graphical Statics*, 26(2), 285--295, 2017.