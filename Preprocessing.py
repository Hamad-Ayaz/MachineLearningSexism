import pandas as pd
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('stopwords')

def remove_noise(text):
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)  # remove links
    text = re.sub('<.*?>', '', text)  # remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    return text

def normalize_case(text):
    return text.lower()

tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

def tokenize(text):
    return tokenizer.tokenize(text)

stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if not word in stop_words]

def process_hashtags_and_mentions(text):
    text = re.sub(r'@\w+', '', text)  # remove mentions
    text = re.sub(r'#', '', text)  # remove hashtag symbol but keep the text
    return text

lemmatizer = WordNetLemmatizer()

def lemmatize(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_tweet(tweet):
    tweet = remove_noise(tweet)
    tweet = normalize_case(tweet)
    tweet = process_hashtags_and_mentions(tweet)
    tokens = tokenize(tweet)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return ' '.join(tokens)  # Convert tokens back to string

# Read data from CSV file
df = pd.read_csv('edos_labelled_data.csv')

# Split the DataFrame into training and test sets based on the 'split' column
train_df = df[df['split'] == 'train'].copy()
test_df = df[df['split'] == 'test'].copy()

# Apply preprocessing to the 'text' column of both the training and test DataFrames
train_df['processed_tweet'] = train_df['text'].apply(preprocess_tweet)
test_df['processed_tweet'] = test_df['text'].apply(preprocess_tweet)

# Print the processed training DataFrame to see the results
print(train_df[['text', 'processed_tweet']])

# Print the processed test DataFrame to see the results
print(test_df[['text', 'processed_tweet']])

# Save the processed training data back into a new CSV file
train_df.to_csv('ProcessedTrainData.csv', index=False)

# Save the processed test data back into a new CSV file
test_df.to_csv('ProcessedTestData.csv', index=False)
