# Data Detective

## Project Description

### Data Detective is an open-source, modular, extensible validation framework for identifying potential issues with heterogeneous, multimodal data.

## Usage Overview

The user must provide a torch dataset describing the datatype of each of its columns with an object
mapping each column to its data type. In order to make use of some of Data Detective’s validators,
the dataset needs to be preprocessed and split, as this allows Data Detective to find mistakes from the preprocessing stage (such as miscoding errors) or the splitting stage (such as splits that are not IID);
however, the user can also provide the raw unsplit dataset if the user does not want to include these
checks. If the user wants to override the default run configuration, they can provide a validation
schema specifying included validations and their options. From there, Data Detective automatically
identifies the relevant validation methods from the selected validators for each datatype and applies
them, returning the results back to the user. For a more detailed usage overview, see the tutorial.

## Questions that are in scope for Data Validator to address
- Do splits used for model training come from othe same distribution?
- Are there any serious outliers present in the dataset?
- Are the conditional independences we expect in the data obeyed?
- Are the datapoints at inference in the same distribution as what we have used to train/test the model?
- Are there any high level "lint-style" issues, including: 
  - weird values being used for empty things, like -999999
  - see data linter

## Installation Steps

```
git clone https://github.com/thelahunginjeet/pyrankagg
git clone https://github.com/thelahunginjeet/kbutil
pip install -r requirements.txt
```

For more information about usage, please see the tutorial.