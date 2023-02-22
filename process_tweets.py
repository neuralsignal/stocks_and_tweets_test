from process_data import load_tweet_data, process_tweets

if __name__ == '__main__':
    print('loading tweets')
    tweet_data = load_tweet_data()
    print('processing')
    tweet_data = process_tweets(tweet_data)
    tweet_data.to_pickle('sentiments.pkl')