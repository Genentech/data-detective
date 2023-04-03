![Data Detective logo](DD_im.png)

# Data Detective

Data Detective is an open-source, modular, extensible validation framework for identifying potential issues with heterogeneous, multimodal data.

## Usage Overview

The user must provide a torch dataset describing the datatype of each of its columns with an object
mapping each column to its data type. In order to make use of some of Data Detective’s validators,
the dataset needs to be preprocessed and split, as this allows Data Detective to find mistakes from the preprocessing stage (such as miscoding errors) or the splitting stage (such as splits that are not IID);
however, the user can also provide the raw unsplit dataset if the user does not want to include these
checks. If the user wants to override the default run configuration, they can provide a validation
schema specifying included validations and their options. From there, Data Detective automatically
identifies the relevant validation methods from the selected validators for each datatype and applies
them, returning the results back to the user. For a more detailed usage overview, see the tutorial.

## Examples of questions that are in scope for Data Validator to address
- Do splits used for model training come from othe same distribution?
- Are there any serious outliers present in the dataset?
- Are the conditional independences we expect in the data obeyed?
- Are the datapoints at inference in the same distribution as what we have used to train/test the model?

## Installation Steps
After cloning the repository and `cd`'ing into the directory, run the following commands. 

```bash
# install packages supporting rank aggregation
git clone https://github.com/thelahunginjeet/pyrankagg.git
git clone https://github.com/thelahunginjeet/kbutil.git

# only if you want DIFFI validator method support 
git clone https://github.com/mattiacarletti/DIFFI.git

# install all other packages
virtualenv dd_venv -p python3.9 
source dd_venv/bin/activate
pip3 install -r requirements.txt
dd_venv/bin/python -m ipykernel install --name=dd_venv 
```

If you are planning on using Data Detective in a jupyter notebook, please ensure that the kernel is switched to the appropriate virtual environoment.

If you are planning to make use of the pretrained transform library for high dimensional inputs, follow the additional install steps outlined below.

```bash
# for huggingface hosted models
pip install transformers

```

# Contributing

To contribute to Data Detective, please first complete the `ExtendingDD` jupyter notebook to learn more about 
how to extend Data Detective to add new validator methods, validators, and validator methods to the Data Detective 
ecosystem. To submit a contribution, issue a pull request containing the contribution as well as any relevant
tests guaranteeing functionality of the implementation. All pull requests must be approved by at least one Data Detective 
administrator before being merged into the master branch. 

There should be at least one test attached to each validator method / transform. All submitted code should be 
well-documented and follow the PEP-8 standard. 
