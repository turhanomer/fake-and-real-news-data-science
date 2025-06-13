import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

print("Loading and preprocessing datasets...")
true_news = pd.read_csv('True.csv')
fake_news = pd.read_csv('Fake.csv')

true_news['label'] = 1
fake_news['label'] = 0

df = pd.concat([true_news, fake_news], ignore_index=True)

print("\nDataset Information:")
print(f"Total number of articles: {len(df)}")
print(f"Number of true news: {len(true_news)}")
print(f"Number of fake news: {len(fake_news)}")

print("\nPreprocessing text data...")
df['content'] = df['title'] + ' ' + df['text']

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stop_words and t.isalnum()]
    return ' '.join(tokens)

df['processed_content'] = df['content'].apply(preprocess_text)

X = df['processed_content']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\nTraining models...")

models = {
    'Naive Bayes': MultinomialNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_vec, y_train)
    pred = model.predict(X_test_vec)
    results[name] = {
        'predictions': pred,
        'report': classification_report(y_test, pred)
    }

print("\nGenerating visualizations...")

plt.figure(figsize=(15, 10))

for i, (name, result) in enumerate(results.items(), 1):
    plt.subplot(2, 2, i)
    cm = confusion_matrix(y_test, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices_enhanced.png')
plt.close()

plt.figure(figsize=(10, 6))
for name, result in results.items():
    y_score = model.predict_proba(X_test_vec)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend(loc="lower right")
plt.savefig('roc_curves.png')
plt.close()

feature_names = vectorizer.get_feature_names_out()
feature_importances = np.abs(models['Random Forest'].feature_importances_)
top_features_idx = np.argsort(feature_importances)[-20:]
top_features = [feature_names[i] for i in top_features_idx]

plt.figure(figsize=(12, 6))
plt.barh(range(len(top_features)), feature_importances[top_features_idx])
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Feature Importance')
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.close()

with open('enhanced_analysis_results.txt', 'w', encoding='utf-8') as f:
    f.write("Enhanced Fake and Real News Dataset Analysis\n")
    f.write("===========================================\n\n")
    
    f.write("1. Dataset Information:\n")
    f.write(f"Total number of articles: {len(df)}\n")
    f.write(f"Number of true news: {len(true_news)}\n")
    f.write(f"Number of fake news: {len(fake_news)}\n\n")
    
    f.write("2. Model Performance:\n")
    for name, result in results.items():
        f.write(f"\n{name} Results:\n")
        f.write(result['report'])
        f.write("\n")

print("\nEnhanced analysis complete! Results have been saved to 'enhanced_analysis_results.txt'")
print("Visualizations have been saved as:")
print("- confusion_matrices_enhanced.png")
print("- roc_curves.png")
print("- feature_distributions.png") 