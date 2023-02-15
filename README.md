# Questions that are in scope for Data Validator to address
- Do splits used for model training come from othe same distribution?
- Are there any serious outliers present in the dataset?
- Are the conditional independences we expect in the data obeyed?
- Are the datapoints at inference in the same distribution as what we have used to train/test the model?
- Are there any high level "lint-style" issues, including: 
  - weird values being used for empty things, like -999999
  - see data linter
