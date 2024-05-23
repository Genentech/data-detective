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

```bash
# install packages supporting rank aggregation
git clone https://github.com/thelahunginjeet/pyrankagg.git
git clone https://github.com/thelahunginjeet/kbutil.git

pip3 install -r requirements.txt
```

[# install all other packages]: #
[virtualenv dd_env -p python3.10]: #
[source dd_env/bin/activate]: #
[pip3 install -r requirements.txt]: #
[dd_env/bin/python -m ipykernel install --name=dd_env]: #

If you are planning on using Data Detective in a jupyter notebook, please ensure that the kernel is switched to the appropriate virtual environoment.

If you are planning to make use of the pretrained transform library for high dimensional inputs, follow the additional install steps outlined below.

```bash
# for huggingface hosted models
pip install transformers
```

## Examples and Guide Notebooks
| notebook | description |
| ----- | ----- |
| Tutorial | To get started on our tutorial dataset and step through each part of an investigation, see the tutorial. |
| Quickstart | To get started as quickly as possible on your own data, please see Quickstart.ipynb in the examples folder. |
| Extending DD | To extend the capacity of Data Detective to your custom validation or transform needs, see ExtendingDD.ipynb |


## Contributing

To contribute to Data Detective, please first complete the `ExtendingDD` jupyter notebook to learn more about 
how to extend Data Detective to add new validator methods, validators, and validator methods to the Data Detective 
ecosystem. To submit a contribution, issue a pull request containing the contribution as well as any relevant
tests guaranteeing functionality of the implementation. All pull requests must be approved by at least one Data Detective 
administrator before being merged into the master branch. 

There should be at least one test attached to each validator method / transform. All submitted code should be 
well-documented and follow the PEP-8 standard. 

## Acknowledgements

- Zhang et. al for KCI/FCIT, used in validator method
- Zhao et. al for pyod
- Kevin Brown for pyrankagg
- all interviewed members of Genentech/Roche for continued feedback during development 
