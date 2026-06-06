import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re

# 1. Load the data
df = pd.read_csv("jud-ipl.csv")

# 2. Clean the text (Remove special characters and lowercase everything)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text) # Removes numbers and symbols
    return text

print("Cleaning case text... this might take a minute...")
df['clean_judgement'] = df['judgement'].apply(clean_text)

# 3. Convert "Accepted/Rejected" into 1s and 0s
le = LabelEncoder()
df['target'] = le.fit_transform(df['label'].astype(str))
# This will likely make 'Accepted' = 0 and 'Rejected' = 1 (or vice versa)

# 4. Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_judgement'], df['target'], test_size=0.20, random_state=42
)

print("\n--- Preprocessing Complete ---")
print(f"Training cases: {len(X_train)}")
print(f"Testing cases: {len(X_test)}")
print(f"Label Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")