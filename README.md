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
│   4045-sentiment-analysis-training.ipynb : training script for models
│   
│   frontend.ipynb : interface for trained model
│   
│   compiled_tweets.csv(input) : input data scraped from internet
│
│   dataset_transformed.csv(input) : data transformed from public datasets
│
│   best_params.tsv(output) : best parameters for the classifier models
│
│   svm_polarity_model.sav(output) : trained svm model for polarity classification
│
│   svm_subjectivity_model.sav(output) : trained svm model for subjectivity classification
│
│   model.png(output) : graphical representation of BERT model
│   
└───bert_polarity_model(output) : fine tuned BERT model for polarity classification
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
└───bert_subjectivity_model(output) : fine tuned BERT model for subjectivity classification
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
└───Dataset(folder) : contains scraping script & dataset transformation