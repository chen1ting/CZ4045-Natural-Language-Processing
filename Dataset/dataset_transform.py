import pandas as pd
from numpy import nan

subjectivityMapping = {
    "Neutral" : 0,
    "Positive" : 1,
    "Negative" : 1,
}
polarityMapping = {
    "Positive" : 1,
    "Negative" : 0,
}

# process dataset 1: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
# this is the main dataset

df_train = pd.read_csv('public_twitter_training.csv', header=None, encoding='latin')
df_test = pd.read_csv('public_twitter_validation.csv', header=None, encoding='latin')

df = pd.concat([df_train, df_test])
df.columns=['Tweet ID', 'Entity','Sentiment', 'Text']
df = df.dropna()
df = df[df['Sentiment'] != 'Irrelevant']
df['WordCnt'] = df['Text'].str.split().str.len()

# this dataset contains duplicate tweet ID, keep the entry with the most word
df = df.sort_values('WordCnt', ascending=False).drop_duplicates('Tweet ID').sort_index()

df["Subjectivity"] = df["Sentiment"].map(subjectivityMapping)
df["Polarity"] = df["Sentiment"].map(polarityMapping)
df = df.drop(columns=['Tweet ID', 'Entity'])

# process dataset 2: https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset
# this dataset is brought in to make sure we have enough objective (neutual tweets) in the dataset

df2 = pd.read_csv('public_twitter_sentiment.csv', encoding='latin')
df2 = df2.rename(columns={'clean_text':'Text', 'category':'Sentiment'})
df2 = df2.dropna()
df2['WordCnt'] = df2['Text'].str.split().str.len()
# select only the tweets with more than 20 words and 0 as sentiment (neutral)
df2 = df2[(df2['WordCnt']>20) & (df2['Sentiment'] == 0)].head(4000).copy()  

df2["Subjectivity"] = df2["Sentiment"]
df2["Polarity"] = nan

# process dataset 3: https://www.kaggle.com/datasets/muhammadiqbalmukati/top10-steam-games-dataset
# this dataset is brought in to make sure we have enough polarity tweets in the dataset

df3 = pd.read_csv('public_steam_sentiment.csv', encoding='latin', usecols=['Review', 'Sentiment'])
df3 = df3.rename(columns={'Review':'Text'})
df3 = df3.dropna()
df3['WordCnt'] = df3['Text'].str.split().str.len()
df3 = df3[df3['WordCnt']>20].copy()  # select only the tweets with more than 20 words
df3 = df3.groupby('Sentiment', group_keys=False).apply(lambda x: x.sample(4000))     # balanced samping data

df3["Subjectivity"] = 1
df3["Polarity"] = df3["Sentiment"]

# output to csv
final_df = pd.concat([df, df2, df3])
final_df.to_csv("dataset_transformed.csv", encoding='latin', index=False)