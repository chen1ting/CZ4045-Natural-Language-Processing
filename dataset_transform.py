import pandas as pd

subjectivityMapping = {
    "Neutral" : 0,
    "Positive" : 1,
    "Negative" : 1,
}
polarityMapping = {
    "Positive" : 1,
    "Negative" : 0,
}

df_train = pd.read_csv('twitter_training.csv', header=None, encoding='latin')
df_test = pd.read_csv('twitter_validation.csv', header=None, encoding='latin')
df = pd.concat([df_train, df_test])
df.columns=['Tweet ID', 'Entity','Sentiment', 'Text']

df["Subjectivity"] = df["Sentiment"].map(subjectivityMapping)
df["Polarity"] = df["Sentiment"].map(polarityMapping)
df['WordCnt'] = df['Text'].str.split().str.len()

# this dataset contains duplicate tweet ID, keep the entry with the most word
df = df.sort_values('WordCnt', ascending=False).drop_duplicates('Tweet ID').sort_index()

df.to_csv("data_transformed.csv", encoding='latin', index=False)