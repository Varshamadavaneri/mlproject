import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load raw CVE dataset
cve_df = pd.read_csv('cve.csv')

# Rename columns
column_mapping = {
    'summary': 'description',
    'cvss': 'cvss_score',
    'pub_date': 'published_date',
    'cwe_name': 'vulnerability_type'
}
cve_df = cve_df.rename(columns=column_mapping)

# Drop rows with missing critical data
cve_df = cve_df.dropna(subset=['description', 'cvss_score'])

# Select useful columns
selected_columns = ['Unnamed: 0', 'description', 'cvss_score', 'published_date', 'vulnerability_type']
cve_df = cve_df[selected_columns]

# Convert CVSS to severity level
def categorize_severity(score):
    try:
        score = float(score)
        if score >= 9.0: return 'Critical'
        elif score >= 7.0: return 'High'
        elif score >= 4.0: return 'Medium'
        else: return 'Low'
    except:
        return 'Unknown'

cve_df['severity'] = cve_df['cvss_score'].apply(categorize_severity)

# Save cleaned dataset
cve_df.to_csv('cleaned_cve.csv', index=False)

# Prepare for training
X = cve_df['description']
y = cve_df['severity']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# Evaluate
y_pred = clf.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save models
joblib.dump(clf, 'severity_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
