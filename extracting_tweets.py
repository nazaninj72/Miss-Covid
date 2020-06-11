import GetOldTweets3 as got
text_query = ['cocaine corona', 'bleach corona', 'alcohol booze corona', 'flu corona', 'lab grown corona']
count = 2000
# Creation of query object
for query in text_query:
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query)\
                                                .setMaxTweets(count)
    # Creation of list that contains all tweets
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    temp = [(tweet.date, tweet.text) for tweet in tweets]
    text_tweets.append(temp)
