![Data Detective logo](DD_im.png)

# Data Detective

Data Detective is an open-source, modular, extensible validation framework for identifying potential issues with heterogeneous, multimodal data.

![image](https://github.com/gred-ecdi/datadetective/assets/97565124/53b70eab-3b38-44e9-bafa-ff660d959f7e)

## Examples of issues that are in scope for Data Detective to detect
- Do splits used for model training come from othe same distribution?
- Are there any anomalies present in the dataset?
- Are the conditional independences we expect in the data obeyed?
- Are the datapoints at inference in the same distribution as what we have used to train/test the model?
- Are there near or exact duplicates present within the dataset?
- Are there mislabeled samples present within the dataset?

## Installation Steps
After cloning the repository and `cd`'ing into the directory, run the following commands. 

```bash
# install packages supporting rank aggregation
git clone https://github.com/thelahunginjeet/pyrankagg.git
git clone https://github.com/thelahunginjeet/kbutil.git

# install all other packages
virtualenv dd_env -p python3.9 
source dd_env/bin/activate
pip3 install -r requirements.txt
dd_env/bin/python -m ipykernel install --name=dd_env 
```

If you are planning on using Data Detective in a jupyter notebook, please ensure that the kernel is switched to the appropriate virtual environoment.

If you are planning to make use of the pretrained transform library for high dimensional inputs, follow the additional install steps outlined below.

```bash
# for huggingface hosted models
pip install transformers

```

## Quickstart

To get started as quickly as possible, please see Quickstart.ipynb in the root directory of this repo.



# Contributing

To contribute to Data Detective, please first complete the `ExtendingDD` jupyter notebook to learn more about 
how to extend Data Detective to add new validator methods, validators, and validator methods to the Data Detective 
ecosystem. To submit a contribution, issue a pull request containing the contribution as well as any relevant
tests guaranteeing functionality of the implementation. All pull requests must be approved by at least one Data Detective 
administrator before being merged into the master branch. 

There should be at least one test attached to each validator method / transform. All submitted code should be 
well-documented and follow the PEP-8 standard. 

### Appendix 1A: Complete list of validator methods

| name | path | method description | data types | operable split types | 
| ---- | ---- | ------------------ | ---------- | -------------------- | 
| adbench_validator_method | src/validator_methods/validator_method_factories/adbench_validator_method_factory.py | factory generating adbench methods that perform anomaly detection | multidimensional data | entire set | 
| adbench_multimodal_validator_method | src/validator_methods/validator_method_factories/adbench_multimodal_validator_method_factory.py | factory generating adbench methods that perform anomaly detection by concatenating all multidimensional columns first to be able to draw conclusions jointly from the data | multidimensional data | entire set | 
| adbench_ood_inference_validator_method | src/validator_methods/validator_method_factories/adbench_ood_inference_validator_method_factory.py | factory generating methods that perform ood testing given a source set and a target/inference set using adbench emthods | multidimensional data | inference_set, everything_but_inference_set | 
| chi square validator method | src/validator_methods/chi_square_validator_method.py | chi square test for testing CI assumptions between two categorical variables | categorical data | entire_set |
| diffi anomaly explanation validator method | src/validator_methods/diffi_anomaly_explanation_validator_method.py | A validator method for explainable anomaly detection using the DIFFI feature importance method. | multidimensional | entire_set |
| fcit validator method | src/validator_methods/fcit_validator_method.py | A method for determining conditionanl independence of two multidimensional vectors given a third. | continuous, categorical, or multidimensional | entire_set |
| kolmogorov smirnov multidimensional split validator | src/validator_methods/kolmogorov_smirnov_multidimensional_split_validator_method.py | KS testing over multidimensional data for split covariate shift. | multidimensional | entire_set |
| kolmogoriv smirnov normality validator method | src/validator_methods/kolmogorov_smirnov_normality_validator_method.py | KS testing over continuous data for normality assumption. | continuous | entire_set | 
| kolmogorov smirnov split validator method | src/validator_methods/kolmogorov_smirnov_split_validator_method.py | KS testing over continuous data for split covariate shift. |  continuous | entire_set |  
| kruskal wallis multidimensional split validator method | src/validator_methods/kruskal_wallis_multidimensional_split_validator_method.py | kruskal wallis testing over multidimensional data for split covariate shift. | multidimensional | entire_set | 
| kruskal wallis split validator method | src/validator_methods/kruskal_wallis_split_validator_method.py | kruskal wallis testing over continuous data for split covariate shift. | continuous | entire_set |  
| mann whitney multidimensional split validator method | src/validator_methods/mann_whitney_multidimensional_split_validator_method.py | mann whitney testing over multidimensional data for split covariate shift. | multidimensional | entire_set |
| mann whitney split validator method | src/validator_methods/mann_whitney_split_validator_method.py | mann whitney testing over continuous data for split covariate shift. | continuous | entire_set |  
| shap tree validator method | src/validator_methods/shap_tree_validator_method.py |     A validator method for explainable anomaly detection using Shapley values. | multidimensional | entire_set | 

