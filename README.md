# 4045NLP

### Setup virtual environment in project

In Windows CMD, ensure you are in the folder of your repository

1. Run `python –m venv venv`
2. Run `venv\Scripts\activate` 
3. Run `pip install -r requirements.txt`

All required packages should have been installed!

`venv\Scripts\activate` is also the <b>command to enter your virtual environment</b> whenever you want to run the application on CMD


### Instruction Guide to read the files

```
project
│   best_params.csv : contains all the trained parameters for symbolic AI
│
│   NLP frontend.ipynb : presentation for model
│   
└───bert_subjectivity_model `trained bert model for subjectivity`
│   │   keras_metadata.pb
│   │   saved_model.pb
│   │
│   └───variables
│   │   │   variables.data-00000-of-00001
│   │   │   variables.index
│   │
│   └───assests
│   │   │   vocab
│   
└───bert_polarity_model `trained bert model for polarity`
│   │   keras_metadata.pb
│   │   saved_model.pb
│   │
│   └───variables
│   │   │   variables.data-00000-of-00001
│   │   │   variables.index
│   │
│   └───assests
│   │   │   vocab
│
└───1. Dataset : contains scraping script & dataset transformation
│   │   file011.txt
│   │   file012.txt
│   
└───2. Model: contains the TRAINING process of models 
│   │   model-training.ipynb  * note: running the training might take long time
│   │   data_transformed.csv  * outside dataset used
│   │   compiled_tweets.csv   * crawled dataset
