import pandas as pd
import numpy as np
import streamlit as st
import nltk
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

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

# Calculate duration in days
df['Days to Close'] = (df['Closed On [...]'] - df['Created Date [...]']).dt.days
df['Days to Close'].fillna(-1, inplace=True)  # Use -1 for tickets that have not been closed yet

# Preprocess 'Name / Summary' for text data
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['Processed_Summary'] = df['Name / Summary'].apply(preprocess_text)

# Encode categorical columns for model training
priority_encoder = LabelEncoder()
severity_encoder = LabelEncoder()
category_encoder = LabelEncoder()

df['Priority_encoded'] = priority_encoder.fit_transform(df['Priority'])
df['Severity_encoded'] = severity_encoder.fit_transform(df['Issue Severity'])
df['Category_encoded'] = category_encoder.fit_transform(df['Category'])

# Define the target variables
y_priority = df['Priority_encoded']
y_severity = df['Severity_encoded']
y_category = df['Category_encoded']
y_duration = df['Days to Close']

# Split the data for each target
X = df[['Processed_Summary']]  # Only use the processed summary for prediction

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_priority, test_size=0.1, random_state=42)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_severity, test_size=0.1, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_category, test_size=0.1, random_state=42)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_duration, test_size=0.1, random_state=42)

# Text vectorizer
text_vectorizer = TfidfVectorizer(max_features=1000)

# Pipeline for predicting Priority
pipeline_priority = Pipeline([
    ('tfidf', text_vectorizer),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Pipeline for predicting Severity
pipeline_severity = Pipeline([
    ('tfidf', text_vectorizer),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Pipeline for predicting Category
pipeline_category = Pipeline([
    ('tfidf', text_vectorizer),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Pipeline for predicting Duration
pipeline_duration = Pipeline([
    ('tfidf', text_vectorizer),
    ('regressor', LinearRegression())
])

# Train the models
pipeline_priority.fit(X_train_p['Processed_Summary'], y_train_p)
pipeline_severity.fit(X_train_s['Processed_Summary'], y_train_s)
pipeline_category.fit(X_train_c['Processed_Summary'], y_train_c)
pipeline_duration.fit(X_train_d['Processed_Summary'], y_train_d)

# Streamlit UI for user interaction
st.title('TicketIQ and Insights')

# Input fields
summary = st.text_input('Name / Summary')

if st.button('Predict'):
    # Preprocess the summary input
    processed_summary = preprocess_text(summary)
    
    # Predict Priority, Severity, Category, and Duration
    predicted_priority_encoded = pipeline_priority.predict([processed_summary])[0]
    predicted_severity_encoded = pipeline_severity.predict([processed_summary])[0]
    predicted_category_encoded = pipeline_category.predict([processed_summary])[0]
    predicted_duration = pipeline_duration.predict([processed_summary])[0]

    # Decode the predictions
    predicted_priority = priority_encoder.inverse_transform([predicted_priority_encoded])[0]
    predicted_severity = severity_encoder.inverse_transform([predicted_severity_encoded])[0]
    predicted_category = category_encoder.inverse_transform([predicted_category_encoded])[0]

    # Calculate Cosine Similarity between the input summary and existing summaries
    vectorized_summaries = text_vectorizer.transform(df['Processed_Summary'])
    input_vector = text_vectorizer.transform([processed_summary])
    
    similarities = cosine_similarity(input_vector, vectorized_summaries).flatten()

    # Get the indices of the 5 most similar tickets
    top_5_similar_indices = similarities.argsort()[-5:][::-1]  # Sort and get top 5 in descending order

    # Retrieve the 5 most similar tickets
    top_5_similar_tickets = df.iloc[top_5_similar_indices]

    # Create a DataFrame for the top 5 similar tickets
    top_5_similar_tickets_df = pd.DataFrame({
        'Ticket Number': top_5_similar_tickets['Number'],
        'Summary': top_5_similar_tickets['Name / Summary'],
        'Priority': top_5_similar_tickets['Priority'],
        'Severity': top_5_similar_tickets['Issue Severity'],
        'Category': top_5_similar_tickets['Category'],
        'Similarity': similarities[top_5_similar_indices],
        'time taken':top_5_similar_tickets['Days to Close']
    })
    
    # Display the top 5 similar tickets in a table
    st.write("### Top 5 Similar Tickets")
    st.table(top_5_similar_tickets_df)
    
    # Create a DataFrame for predicted results
    predicted_results = pd.DataFrame({
        'Predicted Priority': [predicted_priority],
        'Predicted Severity': [predicted_severity],
        'Predicted Category': [predicted_category],
        'Predicted Duration (Days to Close)': [predicted_duration]
    })
    
    # Display the predicted results in a table
    st.write("### Predicted Results")
    st.table(predicted_results)
