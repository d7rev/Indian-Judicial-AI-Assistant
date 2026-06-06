import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load the preprocessed data (assuming you still have the df in memory or reload it)
df = pd.read_csv("jud-ipl.csv").dropna(subset=['judgement', 'label'])
df = df[df['label'].isin(['Accepted', 'Rejected'])] # Let's focus on the clear cases

# 2. Setup the Pipeline: Text -> Numbers -> AI Model
# This counts the importance of words like 'petition' vs common words like 'the'
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
    ('clf', LinearSVC())
])

# 3. Training
print("Training the AI on your MSI Thin 15... please wait...")
model_pipeline.fit(df['judgement'], df['label'])

# 4. Save the "Brain" so you can use it later without retraining
joblib.dump(model_pipeline, 'judicial_model.pkl')
print("\nSuccess! Your model is trained and saved as 'judicial_model.pkl'")