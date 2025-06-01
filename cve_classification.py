import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("cve.csv")

# Rename columns
df = df.rename(columns={
    'summary': 'description',
    'cvss': 'cvss_score',
    'pub_date': 'published_date',
    'cwe_name': 'vulnerability_type'
})

# Drop rows with missing important fields
df = df.dropna(subset=['description', 'cvss_score'])

# Select relevant columns
df = df[['description', 'cvss_score', 'published_date', 'vulnerability_type']]

# Convert CVSS score to severity
def categorize_severity(score):
    try:
        score = float(score)
        if score >= 9.0: return 'Critical'
        elif score >= 7.0: return 'High'
        elif score >= 4.0: return 'Medium'
        else: return 'Low'
    except:
        return 'Unknown'

df['severity'] = df['cvss_score'].apply(categorize_severity)

# Save cleaned data (optional)
df.to_csv("cleaned_cve.csv", index=False)

# Prepare features and labels
X = df['description']
y = df['severity']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# Evaluate
y_pred = clf.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model and components
joblib.dump(clf, 'severity_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
