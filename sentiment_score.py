import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

#VADER (Valence Aware Dictionary and sEntiment Reasoner),
# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Read the CSV file
df = pd.read_csv('data/news-data.csv')
df = df[df['Symbol'] == 'NEPSE']

# Function to calculate sentiment
def get_vader_sentiment(text):
    scores = analyzer.polarity_scores(text)
    sentiment = (
        "Positive" if scores['compound'] > 0.05 else
        "Negative" if scores['compound'] < -0.05 else
        "Neutral"
    )
    return sentiment, scores['compound'], scores['pos'], scores['neu'], scores['neg']

# Apply VADER sentiment analysis to each Title
df[['Sentiment', 'Compound', 'Positive', 'Neutral', 'Negative']] = df['Title'].apply(
    lambda x: pd.Series(get_vader_sentiment(str(x)))
)

# Save the results to a new CSV file
df.to_csv('data/NEPSE_SENTIMENT.csv', index=False)

# Display the first few rows with sentiment scores
print(df.head())
