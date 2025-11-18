import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import wordcloud
from wordcloud import WordCloud

fake_news_df = pd.read_csv('Fake.csv')
real_news_df = pd.read_csv('True.csv')
fake_news_df['label'] = 0
real_news_df['label'] = 1

'''print(real_news_df['text'].value_counts())
print(fake_news_df['text'].value_counts())
print(fake_news_df['text'].describe())
print(real_news_df['text'].describe())'''

#First Visualization: Two Word Clouds for Fake vs Real News Articles
#Basic Cleanning for Word Cloud Visualization
def preprocess_text_case_sensitive(text, custom_stopwords):
    # 1. Tokenize the text (splits into words, keeping case)
    tokens = nltk.word_tokenize(text)
    
    # 2. Filter out stopwords and the specific lowercase 's'
    # The check 'word.lower() not in custom_stopwords' keeps words like 'News'
    # The check 'word != "s"' specifically removes the single lowercase s
    # We also check for single quotes/possessives that might be left by the tokenizer
    filtered_tokens = [
        word for word in tokens 
        if word.lower() not in custom_stopwords 
        and word != 's'
        and word != "'"
        and word != "’"
    ]
    
    # 3. Join the tokens back into a single string for WordCloud
    return ' '.join(filtered_tokens)

custom_stopwords = set(stopwords.words('english'))
custom_stopwords.add('s')
word_pattern = r"\b[a-zA-Z0-9]+\b"

fake_text_processed = preprocess_text_case_sensitive(fake_news_df['text'], custom_stopwords)
real_text_processed = preprocess_text_case_sensitive(real_news_df['text'], custom_stopwords)

fake_text = ' '.join(fake_news_df['text'].astype(str).tolist())
real_text = ' '.join(real_news_df['text'].astype(str).tolist())
fake_wordcloud = WordCloud(width=1200, height=600, background_color='white', stopwords=custom_stopwords,
                           regexp=word_pattern).generate(fake_text_processed)
real_wordcloud = WordCloud(width=1200, height=600, background_color='white', stopwords=custom_stopwords, 
                           regexp=word_pattern).generate(real_text_processed)
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(fake_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Fake News Articles')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(real_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Real News Articles')
plt.axis('off')
plt.show()

#Second Visualization: Visualizing average Word Count Density in Real vs Fake datasets
fake_news_df['word_count'] = fake_news_df['text'].astype(str).apply(lambda x: len(x.split()))
real_news_df['word_count'] = real_news_df['text'].astype(str).apply(lambda x: len(x.split())) 
plt.figure(figsize=(12, 6))
sns.kdeplot(fake_news_df['word_count'], label='Fake News', color='red', fill=True, alpha=0.5)
sns.kdeplot(real_news_df['word_count'], label='Real News', color='blue', fill=True, alpha=0.5)
plt.title('Word Count Distribution in Fake vs Real News Articles')
plt.xlabel('Word Count')
plt.ylabel('Density')
plt.legend()
plt.xlim(0, 2000)  # Limit x-axis for better visualization
plt.grid()
plt.show()
