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
warnings.filterwarnings('ignore')

# YouTube Data API key
API_KEY = "AIzaSyDYT6ezX34KSBzJY3yiT-jqPRI_W6BvsQU"
youtube = build("youtube", "v3", developerKey=API_KEY)

# Taking input from the user and slicing for video id
def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None



def fetch_comments(video_id):

    # Fetch comments
    print("Fetching Comments...")
    comments = []
    nextPageToken = None
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
    return comments



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



# Load the Longformer sentiment pipeline (4096 token capacity)
sentiment_pipeline = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

def split_text(text, max_length=500):
    """Split text into chunks of max_length tokens."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def detect_average_sentiment(comments):
    sentiment_totals = defaultdict(int)
    total_comments = 0

    # Process each comment
    for comment in comments:
        try:
            # Split long comments into smaller chunks
            chunks = split_text(comment, max_length=500)
            chunk_sentiments = []

            # Process each chunk
            for chunk in chunks:
                result = sentiment_pipeline(chunk)
                label = result[0]['label'].lower()
                chunk_sentiments.append(label)

            # Aggregate chunk sentiments (e.g., majority voting)
            final_sentiment = max(set(chunk_sentiments), key=chunk_sentiments.count)
            sentiment_totals[final_sentiment] += 1
            total_comments += 1

        except Exception as e:
            print(f"Skipping a comment due to error: {e}")

    # Calculate average sentiment
    average_sentiment = {
        sentiment: round((count / total_comments) * 100, 2)
        for sentiment, count in sentiment_totals.items()
    }

    return average_sentiment




# Load emotion model
def load_emotion_model():
    model_name = "SamLowe/roberta-base-go_emotions"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def analyze_emotions(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()
        labels = model.config.id2label
        return {labels[i]: float(scores[i]) for i in range(len(scores))}

    return analyze_emotions

# Analyze emotions in batches
def analyze_comments(comments):
    emotion_fn = load_emotion_model()
    all_emotions = defaultdict(list)

    for comment in comments:
        emotions = emotion_fn(comment)
        for emotion, score in emotions.items():
            all_emotions[emotion].append(score)

    # Aggregate average emotions
    avg_emotions = {emotion: np.mean(scores) for emotion, scores in all_emotions.items()}
    return avg_emotions

# Visualize the emotions
def visualize_emotions(emotions):
    plt.figure(figsize=(12, 8))
    emotions = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))
    sns.barplot(x=list(emotions.values()), y=list(emotions.keys()), palette='viridis')
    plt.title("Emotion Analysis of YouTube Comments")
    plt.show()



# Load spaCy's English NLP model
nlp = spacy.load("en_core_web_sm")
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



# Load spaCy's English NLP model
nlp = spacy.load("en_core_web_sm")

def extract_content_ideas(comments, top_n=5):
    """
    Extracts content ideas from a list of comments using NLP.

    Args:
        comments (list): A list of comment strings.
        top_n (int): Number of top content ideas to return.

    Returns:
        list: A list of suggested content ideas.
    """
    idea_keywords = []

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



def get_video_statistics(youtube, video_id):
    """
    Fetches video statistics from YouTube API.

    Args:
        youtube (Resource): YouTube API client object.
        video_id (str): The ID of the YouTube video.

    Returns:
        dict: Dictionary containing video statistics (views, likes, etc.).
    """
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



def get_channel_info(youtube, channel_id):
    """
    Fetches channel details such as title, logo, subscriber count, and more.

    Args:
        youtube (Resource): YouTube API client object.
        channel_id (str): The ID of the YouTube channel.

    Returns:
        dict: Dictionary containing channel details.
    """
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


from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Function to extract relevant words
def extract_relevant_words(comments):
    relevant_words = []

    for comment in comments:
        doc = nlp(comment)
        # Extract entities (people, organizations, products, etc.)
        for ent in doc.ents:
            relevant_words.append(ent.text.lower())

        # Extract noun chunks (main subjects of sentences)
        for chunk in doc.noun_chunks:
            relevant_words.append(chunk.text.lower())

    return relevant_words

# Function to generate a word cloud
def generate_word_cloud(comments):
    # Extract the relevant words
    relevant_words = extract_relevant_words(comments)

    # Join all the words as a single text
    text = " ".join(relevant_words)

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Most Relevant Words from YouTube Comments")
    plt.show()








# Main Processing Function
def process_youtube_video(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        return {"error": "Invalid YouTube URL"}

    try:
        video_response = youtube.videos().list(part='snippet', id=video_id).execute()
        if not video_response.get('items'):
            return {"error": "Video not found"}
        uploader_channel_id = video_response['items'][0]['snippet']['channelId']
    except HttpError as e:
        return {"error": f"API error: {str(e)}"}

    comments = fetch_comments(video_id, uploader_channel_id)
    if isinstance(comments, dict) and "error" in comments:
        return comments

    relevant_comments = [c for c in (filter_comment(c) for c in comments) if c]
    max_emotion_comments = 50  # Limit for performance

    return {
        "video_id": video_id,
        "total_comments": len(comments),
        "relevant_comments": len(relevant_comments),
        "sentiment": analyze_sentiment_advanced(relevant_comments),
        "emotions": analyze_comments(relevant_comments[:max_emotion_comments]),
        "questions": extract_questions(comments),
        "content_ideas": extract_content_ideas(comments),
        "video_statistics": get_video_statistics(youtube, video_id),
        "channel_info": get_channel_info(youtube, uploader_channel_id)
    }
