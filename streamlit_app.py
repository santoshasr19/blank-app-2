import pandas as pd 
import numpy as np
import streamlit as st
import nltk
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer

# Ensure necessary NLTK data files are downloaded
nltk.download('stopwords')

# Load the data from the Excel file
df = pd.read_excel('Upgrade_Defects.xlsm')  # Replace with your actual file path


# Ensure necessary columns are not missing and fill NaNs where needed
df.dropna(subset=['Name / Summary', 'Priority', 'Issue Severity', 'Category', 'Created Date [...]', 'Closed On [...]'], inplace=True)

# Fill missing values
columns_to_fill_unknown = ['Status', 'Priority', 'Category', 'Product', 'Created By [...]', 'Account [...]', 'Targeted to [...]', 
                           'Assigned To [...]', 'Regression', 'Reported In [...]']
for column in columns_to_fill_unknown:
    df[column].fillna('Unknown', inplace=True)

# Handle 'Closed' as False if not known
df['Closed'] = df['Closed'].fillna(False)

# Define the starting date in Excel's serial number format (January 1, 1900)
excel_start_date = pd.Timestamp('1899-12-30')

# Convert serial numbers to actual dates
df['Created Date [...]'] = excel_start_date + pd.to_timedelta(df['Created Date [...]'], unit='D')
df['Closed On [...]'] = excel_start_date + pd.to_timedelta(df['Closed On [...]'], unit='D')


# # Convert dates to datetime
# df['Created Date [...]'] = pd.to_datetime(df['Created Date [...]'], errors='coerce')
# df['Closed On [...]'] = pd.to_datetime(df['Closed On [...]'], errors='coerce')

print('Created Date [...]')
print(df['Created Date [...]'])
print("closed on")
print(df['Closed On [...]'])

# Calculate duration in days
df['Days to Close'] = (df['Closed On [...]'] - df['Created Date [...]']).dt.days
df['Days to Close'].fillna(-1, inplace=True)  # Use -1 for tickets that have not been closed yet

# Handle 'Issue Severity' missing values
df['Issue Severity'] = df['Issue Severity'].fillna('Unknown')

# Preprocess 'Name / Summary' for text data
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['Processed_Summary'] = df['Name / Summary'].apply(preprocess_text)

# Calculate the difference in days between Created Date and Closed On
df['Days to Close'] = (df['Closed On [...]'] - df['Created Date [...]']).dt.days

print(df['Closed On [...]'])
print(df['Created Date [...]'])


# Combine features for models
df['Combined_Feature'] = df['Processed_Summary'] + ' ' + df['Issue Severity']

# Prepare data for regression
X = df[['Combined_Feature', 'Priority', 'Issue Severity']]
y_duration = df['Days to Close']
print(y_duration)

# Train-test split
X_train, X_test, y_duration_train, y_duration_test = train_test_split(X, y_duration, test_size=0.1, random_state=42)

# One-hot encode 'Priority' and 'Issue Severity'
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

# Define preprocessor for text features
text_preprocessor = TfidfVectorizer(max_features=1000)

# Create pipeline for duration prediction
pipeline_duration = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('text', text_preprocessor, 'Combined_Feature'),
        ('onehot', one_hot_encoder, ['Priority', 'Issue Severity'])
    ])),
    ('regressor', LinearRegression())
])

# Train the model
pipeline_duration.fit(X_train, y_duration_train)

# Evaluate the model
print("Duration Model R^2 Score:", pipeline_duration.score(X_test, y_duration_test))

# Vectorize the 'Processed_Summary' using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Processed_Summary'])

# Function to find similar tickets
def find_similar_ticket(new_ticket_summary, df, tfidf_vectorizer, tfidf_matrix, threshold=0.7):
    new_ticket_processed = preprocess_text(new_ticket_summary)
    new_ticket_vector = tfidf_vectorizer.transform([new_ticket_processed])
    similarity_scores = cosine_similarity(new_ticket_vector, tfidf_matrix)
    similar_ticket_indices = similarity_scores[0].argsort()[::-1]

    similar_tickets = []
    for idx in similar_ticket_indices:
        if similarity_scores[0][idx] >= threshold:
            similar_tickets.append((df.iloc[idx]['Number'], df.iloc[idx]['Name / Summary'], similarity_scores[0][idx]))
        else:
            break
    return similar_tickets

# Prediction functions
def predict_duration(summary, priority, severity):
    """Predict duration (time required to resolve the ticket) based on summary, priority, and severity."""
    features = pd.DataFrame([[summary, priority, severity]], columns=['Combined_Feature', 'Priority', 'Issue Severity'])
    return pipeline_duration.predict(features)[0]

# Streamlit UI for user interaction
st.title('Ticket Prediction System')

# Input fields
summary = st.text_input('Name / Summary')
priority = st.selectbox('Priority', ['1', '2', '3', 'Blocking', 'Unknown'])
severity = st.selectbox('Severity', ['Critical', 'Major', 'Minor', 'Unknown'])

if st.button('Predict'):
    # Find similar tickets
    similar_tickets = find_similar_ticket(summary, df, tfidf_vectorizer, tfidf_matrix)
    st.write("Similar Tickets:")
    for ticket in similar_tickets:
        st.write(f"Ticket Number: {ticket[0]}, Summary: {ticket[1]}, Similarity: {ticket[2]:.2f}")
    # Make predictions
    predicted_duration = predict_duration(summary, priority, severity)

    # Display results
    st.write(f"Duration (Days to Close): {predicted_duration:.2f}")
