**Final Flask Project DataRoot**


The aim of this project is to provide developing a machine learning solution from start to finish. By following data preprocessing, feature engineering, model selection and training, web API was build.

Dataset wat taken from Kaggle:

https://www.kaggle.com/jsphyg/weather-dataset-rattle-package

Repository (image) on Docker Hub:
```
oleksii1994/flask_project:v9
```

```
app
    ├── data                        - contains train and validation data
    │   ├── train.csv               - train set 
    │   └── val.csv                 - validation set (must contain target values)
    ├── models                      - this folder contains a trained estimator.
    │   └── <name>.pickle           - trained estimator. 
    │
    ├── settings                    - here you can store different constant values, connection parameters, etc.
    │   ├── constants.py            - multiple constants storage for their convenient usage.
    │   └── specifications.json     - specifications of your data preprocessing operations.   
    │   
    ├── utils                       - this folder contains instruments we'll use to work with dataset.
    │   ├── __init__.py             - init file for the package. 
    │   ├── dataloader.py           - dataloader. 
    │   ├── dataset.py              - class dedicated for giving info about the dataset.
    │   ├── predictor.py            - predictor.
    │   ├── save_model.py           - save model.
    │   ├── test_data.py            - test model on VAL.CSV
    │   ├── test_different_models.py- choose the best estimator
    │   └── trainer.py              - train script.
    │ 
    ├── app.py                      - route, app.
    │
    ├── final_check.py              - test api and model
    │
    ├── rain_australia.ipynb        - colab notebook
    │
    ├── requirements.txt			- list of libraries used for Dockerization 
    │
    └── Dockerfile					- commands used for Dockerization

```