import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("Slogans.csv")

# Selecting relevant columns
df = df[['Slogan', 'Raw_Revenue_Info', 'identical_trademarks', 'simillar_trademarks', 'social_media_taken', 'domains_point_to_website']]
df = df.dropna()

# Convert categorical features into numeric
for col in ['Raw_Revenue_Info', 'identical_trademarks', 'simillar_trademarks', 'social_media_taken', 'domains_point_to_website']:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert non-numeric to NaN
df = df.fillna(0)  # Replace NaN with 0

# Define features (X) and target (y)
X_text = df['Slogan']  # Text feature (slogan)
X_numeric = df[['Raw_Revenue_Info', 'identical_trademarks', 'simillar_trademarks', 'social_media_taken']].values  # Numeric features
y = df['domains_point_to_website'].values  # Target label (continuous score)

# Text feature extraction using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_text_transformed = vectorizer.fit_transform(X_text)

# Scaling the numeric features
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# Combine text and numeric features
X = np.hstack((X_text_transformed.toarray(), X_numeric_scaled))

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection - Linear Regression
model = Ridge(alpha=1.0)  # You can tune alpha
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Save the trained model and vectorizer
joblib.dump(model, 'slogan_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(scaler, 'scaler.pkl')
