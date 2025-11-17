import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import make_pipeline
import joblib
import os

# Load your dataset
df = pd.read_csv('news_articles.csv')

# Clean & combine title + text (this gives best accuracy in fake news tasks)
df['content'] = (df['title'].fillna('') + " " + df['text'].fillna('')).str.lower()

# Keep only English articles and valid labels
df = df[df['language'] == 'english']
df = df[df['label'].isin(['Real', 'Fake'])]
df = df[['content', 'label']]

# Train on ALL data (max accuracy) - PassiveAggressive is best for this dataset
pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=50000, sublinear_tf=True),
    PassiveAggressiveClassifier(max_iter=1000, random_state=42, C=0.1)
)

pipeline.fit(df['content'], df['label'])

# Save model
joblib.dump(pipeline, 'model.pkl')
print("Model trained and saved as model.pkl (size ~150-300MB)")
