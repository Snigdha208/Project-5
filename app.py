import pandas as pd
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer  # ✅ Add this line


# Load model and vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load preprocessed dataset with true sentiments
df = pd.read_csv(r'F:\vscode\AI ECHO\chatgpt_reviews_processed.csv')

st.title("Sentiment Analysis with Comparison")

user_input = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text for prediction.")
    else:
        # Predict sentiment
        input_tfidf = tfidf.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        pred_sentiment = sentiment_map.get(prediction, str(prediction))
        
        # Display the prediction
        st.write(f"**Predicted Sentiment:** {pred_sentiment}")

st.markdown("---")
st.header("Sentiment Analysis Insights")

# 1. Overall sentiment proportions
with st.expander("1. What is the overall sentiment of user reviews?"):
    sentiment_counts = df['sentiment'].value_counts(normalize=True).round(2) * 100
    st.bar_chart(sentiment_counts)

# 2. Sentiment by rating
with st.expander("2. How does sentiment vary by rating?"):
    if 'rating' in df.columns:
        st.bar_chart(df.groupby('rating')['sentiment'].value_counts(normalize=True).unstack().fillna(0))
    else:
        st.info("Rating column not found.")

# 3. Keywords/phrases by sentiment
with st.expander("3. Which keywords are most associated with each sentiment class?"):
    for sentiment in ['positive', 'neutral', 'negative']:
        st.write(f"**{sentiment.capitalize()} reviews word cloud**")
        text = ' '.join(df[df['sentiment'] == sentiment]['review'].astype(str))
        wc = WordCloud(width=600, height=400, background_color='white').generate(text)
        st.image(wc.to_array())

# 4. Sentiment over time
with st.expander("4. How has sentiment changed over time?"):
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df_time = df.dropna(subset=['date'])
        sentiment_trend = df_time.groupby(df_time['date'].dt.to_period('M'))['sentiment'].value_counts().unstack().fillna(0)
        sentiment_trend.index = sentiment_trend.index.to_timestamp()
        st.line_chart(sentiment_trend)
    else:
        st.info("Date column not found.")

# 5. Sentiment vs. verified users
with st.expander("5. Do verified users tend to leave more positive or negative reviews?"):
    if 'verified_purchase' in df.columns:
        st.bar_chart(df.groupby('verified_purchase')['sentiment'].value_counts(normalize=True).unstack().fillna(0))
    else:
        st.info("Verified purchase column not found.")

# 6. Sentiment vs. review length
with st.expander("6. Are longer reviews more likely to be negative or positive?"):
    df['review_length'] = df['review'].astype(str).apply(len)
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='sentiment', y='review_length', ax=ax)
    st.pyplot(fig)

# 7. Sentiment by location
with st.expander("7. Which locations show the most positive or negative sentiment?"):
    if 'location' in df.columns:
        st.bar_chart(df.groupby('location')['sentiment'].value_counts(normalize=True).unstack().fillna(0))
    else:
        st.info("Location column not found.")

# 8. Sentiment across platforms
with st.expander("8. Is there a difference in sentiment across platforms?"):
    if 'platform' in df.columns:
        st.bar_chart(df.groupby('platform')['sentiment'].value_counts(normalize=True).unstack().fillna(0))
    else:
        st.info("Platform column not found.")

# 9. Sentiment by ChatGPT version
with st.expander("9. Which ChatGPT versions are associated with higher/lower sentiment?"):
    if 'version' in df.columns:
        st.bar_chart(df.groupby('version')['sentiment'].value_counts(normalize=True).unstack().fillna(0))
    else:
        st.info("ChatGPT version column not found.")

# 10. Common themes in negative reviews
with st.expander("10. What are the most common negative feedback themes?"):
    negative_reviews = df[df['sentiment'] == 'negative']['review'].astype(str)
    vectorizer = CountVectorizer(stop_words='english', max_features=20)
    X = vectorizer.fit_transform(negative_reviews)
    keywords = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()).sum().sort_values(ascending=False)
    st.bar_chart(keywords)