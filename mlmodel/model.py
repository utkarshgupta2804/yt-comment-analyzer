# For Fetching Comments
from googleapiclient.discovery import build
# For filtering comments
import re
# For filtering comments with just emojis
import emoji
# Analyze the sentiments of the comment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# For visualization
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import spacy
from collections import Counter
from googleapiclient.errors import HttpError
from collections import defaultdict
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import json
import numpy as np
import warnings
import joblib
warnings.filterwarnings('ignore')

# File paths
URL_JSON_PATH = "url.json"
RESULTS_JSON_PATH = "analysis.json"

# YouTube Data API key
API_KEY = "AIzaSyDYT6ezX34KSBzJY3yiT-jqPRI_W6BvsQU"
youtube = build("youtube", "v3", developerKey=API_KEY)

# Taking input from the user and slicing for video id
def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

# Read URL from JSON file
try:
    with open(URL_JSON_PATH, 'r') as f:
        data = json.load(f)
        video_url = data.get("url")
        if not video_url:
            print("No URL found in JSON file")
            exit()
except Exception as e:
    print(f"Error reading URL from JSON: {str(e)}")
    exit()

video_id = extract_video_id(video_url)

if not video_id:
    print("Invalid YouTube URL. Please enter a correct video URL.")
    # Save error to results JSON
    with open(RESULTS_JSON_PATH, 'w') as f:
        json.dump({"error": "Invalid YouTube URL"}, f)
    exit()

print("Extracted video ID:", video_id)

# Getting the channelId of the video uploader
try:
    video_response = youtube.videos().list(
        part='snippet',
        id=video_id
    ).execute()
    
    # Splitting the response for channelID
    video_snippet = video_response['items'][0]['snippet']
    uploader_channel_id = video_snippet['channelId']
    print("channel id: " + uploader_channel_id)
except Exception as e:
    print(f"Error getting video info: {str(e)}")
    # Save error to results JSON
    with open(RESULTS_JSON_PATH, 'w') as f:
        json.dump({"error": f"Error getting video info: {str(e)}"}, f)
    exit()

def fetch_comments(video_id):
    # Fetch comments
    print("Fetching Comments...")
    comments = []
    nextPageToken = None
    try:
        while True:
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,  # You can fetch up to 100 comments per request
                pageToken=nextPageToken
            )
            response = request.execute()
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                # Check if the comment is not from the video uploader
                if comment['authorChannelId']['value'] != uploader_channel_id:
                    comments.append(comment['textDisplay'])
            nextPageToken = response.get('nextPageToken')

            if not nextPageToken:
                break
    except Exception as e:
        print(f"Error fetching comments: {str(e)}")
    return comments

comments = fetch_comments(video_id)
print(f"Total comments fetched: {len(comments)}")

# Constants
MAX_LENGTH = 500  # Maximum comment length
THRESHOLD_RATIO = 0.7  # Text-to-emoji ratio threshold
hyperlink_pattern = re.compile(r'https?://\S+|www\.\S+')  # Regex to detect links

def is_spam(comment):
    """
    Detects if a comment is spam based on repeated words, excessive links, or special characters.
    """
    words = comment.lower().split()
    word_counts = Counter(words)

    # Condition 1: More than 50% of words are repeated
    if any(count > len(words) / 2 for count in word_counts.values()):
        return True

    # Condition 2: More than 2 links
    if len(re.findall(hyperlink_pattern, comment)) > 2:
        return True

    # Condition 3: Excessive special characters (more than 30% of the comment)
    if len(re.findall(r'[^a-zA-Z0-9\s]', comment)) > len(comment) * 0.3:
        return True

    return False

def filter_comment(comment):
    """Filters a single comment based on hyperlink presence, emoji ratio, duplicates, and length."""
    comment = comment.lower().strip()  # Normalize comment

    # Count emojis and non-space text characters
    emojis = emoji.emoji_count(comment)
    text_characters = len(re.sub(r'\s', '', comment))

    # Remove hyperlinks
    if hyperlink_pattern.search(comment):
        return None  # Discard comments with links

    # Ensure the comment has actual text (not just emojis/spam)
    if not any(char.isalnum() for char in comment):
        return None

    # Check emoji ratio (avoid spammy comments with too many emojis)
    if emojis > 0 and (text_characters / (text_characters + emojis)) < THRESHOLD_RATIO:
        return None

    # Remove excessively long comments (e.g., more than 300 characters)
    if len(comment.split()) > 500:
        return None

    return comment  # Return the cleaned comment as a string

# Apply the filter to all comments
relevant_comments = [comment for comment in (filter_comment(c) for c in comments) if comment]
print(f"Total relevant comments: {len(relevant_comments)}")

# Save comments to file
with open("ytcomments.txt", 'w', encoding='utf-8') as f:
    for comment in relevant_comments:
        if isinstance(comment, list):  # If comment is a list, join elements into a single string
            comment = " ".join(str(c) for c in comment)  # Ensure all elements are strings
        if comment:  # Avoid writing empty or None values
            f.write(comment + "\n")

from transformers import pipeline, AutoTokenizer

def analyze_sentiment_advanced(comments):
    # Load a pre-trained sentiment analysis model
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # Load the tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    # Initialize counters
    positive = 0
    negative = 0
    neutral = 0
    dropped_comments = 0  # Track how many comments are dropped

    # Analyze each comment
    for comment in comments:
        # Tokenize the comment to check its length
        tokens = tokenizer.encode(comment, truncation=False)

        # Drop comments that exceed 512 tokens
        if len(tokens) > 512:
            dropped_comments += 1
            continue  # Skip this comment

        # Use the pipeline to get sentiment and score
        result = sentiment_pipeline(comment)[0]
        label = result['label']
        score = result['score']

        # Classify based on the label
        if label == "POSITIVE":
            positive += 1
        elif label == "NEGATIVE":
            negative += 1
        else:
            neutral += 1

    # Calculate percentages
    total_comments = len(comments) - dropped_comments  # Exclude dropped comments
    if total_comments > 0:  # Avoid division by zero
        positive_percentage = (positive / total_comments) * 100
        negative_percentage = (negative / total_comments) * 100
        neutral_percentage = (neutral / total_comments) * 100
    else:
        positive_percentage = 0
        negative_percentage = 0
        neutral_percentage = 0

    return {
        "positive": positive_percentage,
        "negative": negative_percentage,
        "neutral": neutral_percentage,
        "dropped_comments": dropped_comments  # Return the number of dropped comments
    }

sentiment_result = analyze_sentiment_advanced(relevant_comments)
print(sentiment_result)

# Get video statistics
def get_video_statistics(youtube, video_id):
    try:
        response = youtube.videos().list(
            part="statistics",
            id=video_id
        ).execute()

        if "items" in response and response["items"]:
            return response["items"][0].get("statistics", {})
        else:
            print("No statistics found for this video.")
            return {}

    except HttpError as error:
        print(f"An error occurred: {error}")
        return {}

video_stats = get_video_statistics(youtube, video_id)

# Get channel info
def get_channel_info(youtube, channel_id):
    try:
        response = youtube.channels().list(
            part="snippet,statistics,brandingSettings",
            id=channel_id
        ).execute()

        if "items" in response and response["items"]:
            channel_data = response["items"][0]

            return {
                "channel_title": channel_data["snippet"].get("title", "Unknown"),
                "video_count": channel_data["statistics"].get("videoCount", "N/A"),
                "channel_logo_url": channel_data["snippet"]["thumbnails"]["high"].get("url", ""),
                "channel_created_date": channel_data["snippet"].get("publishedAt", "N/A"),
                "subscriber_count": channel_data["statistics"].get("subscriberCount", "N/A"),
                "channel_description": channel_data["snippet"].get("description", "No description available."),
            }
        else:
            print("No channel information found.")
            return {}

    except HttpError as error:
        print(f"An error occurred: {error}")
        return {}

channel_info = get_channel_info(youtube, uploader_channel_id)

# Extract content ideas
def extract_content_ideas(comments, top_n=5):
    """
    Extracts content ideas from a list of comments using NLP.
    """
    idea_keywords = []
    nlp = spacy.load("en_core_web_sm")

    for comment in comments:
        doc = nlp(comment.lower())  # Process text
        # Extract noun chunks (phrases representing topics)
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip()
            # Ignore very short phrases and stopwords
            if len(phrase) > 2 and not re.match(r'\b(is|the|a|an|this|that|it|you|I|we|they)\b', phrase):
                idea_keywords.append(phrase)

    # Count frequency of extracted phrases
    common_ideas = Counter(idea_keywords).most_common(top_n)

    # Format as content ideas
    content_ideas = [idea[0] for idea in common_ideas]

    return content_ideas

content_ideas = extract_content_ideas(relevant_comments)

# Extract questions
def extract_questions(comments):
    """
    Extract questions from a list of comments based on context (not just keywords).
    """
    nlp = spacy.load("en_core_web_sm")

    questions = []
    question_phrases = ["can you", "how do", "is it", "do you think", "would it be", "what if", "why is"]

    for comment in comments:
        doc = nlp(comment)

        # Check if the comment seems like a question based on structure
        for sent in doc.sents:
            sent_lower = sent.text.lower()

            # 1. Check if there's a question phrase OR context seems like a question
            if any(phrase in sent_lower for phrase in question_phrases):
                questions.append(sent.text)
                continue

            # 2. Check if the sentence structure has a question pattern
            if sent.text.endswith('?'):
                questions.append(sent.text)
                continue

            # 3. Predict if it 'feels like' a question (based on auxiliary verbs)
            if any(token.tag_ in ["MD"] for token in sent):  # MD = Modal Verb (can, could, would, should, etc.)
                questions.append(sent.text)
                continue

    return questions[:5]

extracted_questions = extract_questions(comments)

# Prepare the results JSON
results = {
    "video_id": video_id,
    "total_comments": len(relevant_comments),
    "sentiment": sentiment_result,
    "video_statistics": video_stats,
    "channel_info": channel_info,
    "content_ideas": content_ideas,
    "common_questions": extracted_questions
}

# Save results to JSON file
with open(RESULTS_JSON_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print("Analysis complete. Results saved to", RESULTS_JSON_PATH)