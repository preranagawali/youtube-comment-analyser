import spacy
from textblob import TextBlob
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from collections import Counter
from wordcloud import WordCloud
from transformers import pipeline

# Load spaCy's small English model
nlp = spacy.load('en_core_web_sm')

# Load emotion analysis model from transformers
emotion_analyzer = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True)

# YouTube API key (hard-coded for now)
API_KEY = "Google-Youtube-API-Key"

# Function to get YouTube comments
def get_youtube_comments(api_key, video_id, max_results=100):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    request = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        maxResults=max_results,
        textFormat='plainText'
    )
    response = request.execute()
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textOriginal']
        comments.append(comment)
    return comments

# Function to preprocess comments using spaCy
def preprocess_comments(comments):
    cleaned_comments = []
    for comment in comments:
        # Remove URLs, mentions, hashtags, and special characters
        comment = re.sub(r"http\S+|www\S+|https\S+", '', comment, flags=re.MULTILINE)
        comment = re.sub(r'\@\w+|\#', '', comment)
        comment = re.sub(r'[^A-Za-z\s]', '', comment)
        comment = re.sub(r'\d+', '', comment)

        # Process the comment with spaCy
        doc = nlp(comment)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        cleaned_comments.append(' '.join(tokens))
        
    return cleaned_comments

# Function to analyze sentiment using the transformers model
def analyze_sentiment(comments):
    sentiment_results = []
    for comment in comments:
        result = emotion_analyzer(comment)
        dominant_emotion = max(result[0], key=lambda x: x['score'])
        sentiment_results.append(dominant_emotion['label'])
    return sentiment_results

# Function to visualize sentiment results and most used words
def visualize_results(sentiment_results, cleaned_comments):
    sentiment_df = pd.DataFrame(sentiment_results, columns=['sentiment'])

    # Sentiment Distribution - Pie chart
    fig1, ax1 = plt.subplots()
    sentiment_df['sentiment'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette("tab10"), ax=ax1)
    ax1.set_title('Sentiment Distribution')
    ax1.set_ylabel('')

    # Sentiment Counts - Bar chart
    fig2, ax2 = plt.subplots()
    sns.countplot(x='sentiment', data=sentiment_df, palette='tab10', ax=ax2)
    ax2.set_title('Sentiment Counts')
    
    # Word Cloud for most used words
    all_words = ' '.join(cleaned_comments)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.imshow(wordcloud, interpolation='bilinear')
    ax3.axis('off')
    ax3.set_title('Most Used Words')
    
    return fig1, fig2, fig3

# Streamlit app
def main():
    st.set_page_config(page_title="YouTube Comments Sentiment Analysis", layout="wide")
    
    st.title("📊 YouTube Comments Sentiment Analysis")
    st.markdown("Analyze the sentiment of comments on any YouTube video in a visually appealing way!")

    # Sidebar for inputs
    st.sidebar.header("Input Parameters")
    video_url = st.sidebar.text_input("Enter YouTube Video URL")
    max_results = st.sidebar.slider("Number of Comments to Analyze", min_value=100, max_value=1000, value=500)

    if st.sidebar.button("Analyze"):
        if not video_url:
            st.sidebar.error("Please enter the video URL.")
        else:
            video_id = video_url.split("v=")[-1]
            with st.spinner("Fetching comments..."):
                comments = get_youtube_comments(API_KEY, video_id, max_results)
            with st.spinner("Cleaning comments..."):
                cleaned_comments = preprocess_comments(comments)
            with st.spinner("Analyzing sentiment..."):
                sentiment_results = analyze_sentiment(cleaned_comments)
            st.success("Analysis complete!")

            st.markdown("### Sentiment Analysis Results")
            fig1, fig2, fig3 = visualize_results(sentiment_results, cleaned_comments)
            st.pyplot(fig1)
            st.pyplot(fig2)
            st.pyplot(fig3)

if __name__ == "__main__":
    main()
