import numpy as np
import pandas as pd
import json
import datetime

from pathlib import Path


STOCKPATH = Path("stocknet-dataset/price/preprocessed")
TWITTERPATH = Path("stocknet-dataset/tweet/preprocessed")

STOCK_COLS = ['mvt', 'open', 'high', 'low', 'close']
TWITTER_COLS = ['text', 'counts']
SHARED_COLS = ['date', 'stock_name']


def load_stock_data(stockpath=STOCKPATH):
    stock_data = []
    for istock in stockpath.glob('*.txt'):
        data = pd.read_table(istock, header=None, names=['date', 'mvt', 'open', 'high', 'low', 'close', 'vol'])
        data['date'] = pd.to_datetime(data['date'])
        data['stock_name'] = istock.stem
        stock_data.append(data)
        
    stock_data = pd.concat(stock_data)
    stock_data['date'] = stock_data['date'].dt.date
    return stock_data.sort_values('date').reset_index(drop=True)


def smooth_stock_data(
    stock_data, 
    halflifes=[2.5, 3.75, 5, 7.5, 10, 20], 
    cols=STOCK_COLS
):
    # create exponentially smoothed versions of the stock movements using prior time points
    all_stock_data = stock_data
    for idx, halflife in enumerate(halflifes):
        smooths = []
        for name, df in stock_data.groupby('stock_name'):
            smooth = df[cols].ewm(
                halflife=f'{halflife} days', times=pd.DatetimeIndex(df['date'])
            ).mean()
            smooth['date'] = df['date']
            smooth['stock_name'] = name
            smooths.append(smooth)
        smooths = pd.concat(smooths)
        smooths = smooths.rename(columns={col:f"{col}{idx}" for col in cols})
        all_stock_data = pd.merge(smooths, all_stock_data, on=['date', 'stock_name'])
    return all_stock_data


def label_stocks(stock_data, bottom=-0.005, top=0.0055):
    # stocks must be sorted by dates
    minus = stock_data['mvt'] <= bottom
    plus = stock_data['mvt'] > top
    neutral = ~minus & ~plus
    mvt = stock_data['mvt']
    
    stock_data = stock_data.iloc[:-1].copy()
    stock_data['minus'] = minus.to_numpy()[1:]
    stock_data['plus'] = plus.to_numpy()[1:]
    stock_data['neutral'] = neutral.to_numpy()[1:]
    stock_data['next_mvt'] = mvt.to_numpy()[1:]
    return stock_data


def process_stocks(
    stockpath=STOCKPATH,
    start_date=datetime.date(2013, 12, 31), 
    end_date=datetime.date(2016, 1, 2),
    halflifes=[2.5, 3.75, 5, 7.5, 10, 20], 
    train_stop=datetime.date(2015, 8, 1),
    val_stop=datetime.date(2015, 10, 1),
    bottom=-0.005, top=0.0055, 
    add_sentiments=False
):
    stock_data = load_stock_data(stockpath)
        
    stock_data = stock_data[
        (stock_data['date'] >= start_date) 
        & (stock_data['date'] <= end_date)
    ]
    
    stock_data = smooth_stock_data(stock_data, halflifes=halflifes)
    stock_data = label_stocks(stock_data, bottom, top)
    
    if add_sentiments:
        # only works after running proces_tweets.py
        sentiments = pd.read_pickle('sentiments.pkl')
        sentiments = sentiments[
            (sentiments['date'] >= start_date) 
            & (sentiments['date'] <= end_date)
        ]
        sent_cols = list(set(sentiments.columns) - set(SHARED_COLS))
        sentiments = sentiments.groupby(SHARED_COLS)[sent_cols].mean().reset_index()
        print(sentiments.shape, stock_data.shape)
        stock_data = pd.merge(stock_data, sentiments, how='left', on=SHARED_COLS)
        stock_data[sent_cols] = stock_data[sent_cols].fillna(1/3)
    
    x_cols = list(
        set(stock_data.columns) - set([
            'date', 'minus', 'plus', 
            'neutral', 'stock_name', 'next_mvt'
        ])
    )
    y_cols = ['next_mvt']
    
    stocks_df = stock_data.pivot(
        index='date', columns='stock_name', 
        values=x_cols
    ).fillna(0)
    
    stocks_ydf = stock_data.pivot(
        index='date', columns='stock_name', 
        values=y_cols
    ).fillna(False)
    
    X = stocks_df.to_numpy()
    Y = stocks_ydf.to_numpy()

    train_bool = (stocks_df.index < train_stop)
    val_bool = ~train_bool & (stocks_df.index < val_stop)
    test_bool = ~train_bool & ~val_bool

    Xtrain, Xval, Xtest = X[train_bool], X[val_bool], X[test_bool]
    Ytrain, Yval, Ytest = Y[train_bool], Y[val_bool], Y[test_bool]
    
    return Xtrain, Xval, Xtest, Ytrain, Yval, Ytest


def process_tweet_group(series : pd.Series):
    values, counts = np.unique(series, return_counts=True)
    return pd.Series([values, counts], index=TWITTER_COLS)


def load_tweet_data(twitterpath=TWITTERPATH, group=True):
    tweet_data = []
    for istock in twitterpath.glob('*'):
        for itweet in istock.glob('*'):
            with open(itweet, 'r') as f:
                for l in f.readlines():
                    data = json.loads(l)
                    tweet_data.append(
                        {
                            'text': ' '.join(data['text']), 
                            'user_id': data['user_id_str'], 
                            'stock_name': istock.stem, 
                            'date': pd.to_datetime(data['created_at'])
                        }
                    )
    
    tweet_data = pd.DataFrame(tweet_data)
    if group:
        tweet_data = tweet_data.groupby(
            ['date', 'stock_name']
        )['text'].apply(process_tweet_group).unstack(level=2).reset_index().sort_values('date').reset_index(drop=True)
    tweet_data['datetime'] = tweet_data['date']
    tweet_data['date'] = tweet_data['date'].dt.date
    return tweet_data


def process_tweets(tweet_data, savepath=Path('tweet_sentiments')):
    # create sentiment scores for each tweet
    from transformers import AutoModelForSequenceClassification
    from transformers import AutoTokenizer
    from scipy.special import softmax
    from tqdm import tqdm
    
    savepath.mkdir(exist_ok=True, parents=True)
    MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    model_name = MODEL + '1'
    if Path(model_name).exists():
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    else:
        print('loading and saving model')
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        model.save_pretrained(model_name)
    
    for idx, row in tqdm(tweet_data.iterrows(), total=len(tweet_data)):
        counts = row['counts']
        sentimentpath = savepath / f"{row['datetime']}-{row['stock_name']}.npy"
        
        if sentimentpath.exists():
            outputs = np.load(sentimentpath)
        else:
            sentences = list(row['text'])
            
            inputs = tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True)
            outputs = model(**inputs)
            
            outputs.logits.detach().numpy()
            outputs = softmax(outputs.logits.detach().numpy(), axis=1)
            np.save(sentimentpath, outputs)        
        
        # weighted average
        avg = np.average(outputs, axis=0, weights=counts)
        mean = np.mean(outputs, axis=0)
        
        tweet_data.loc[idx, ['avg0', 'avg1', 'avg2']] = avg
        tweet_data.loc[idx, ['mean0', 'mean1', 'mean2']] = mean
            
    tweet_data = tweet_data[
        SHARED_COLS+['avg0', 'avg1', 'avg2', 'mean0', 'mean1', 'mean2']
    ].copy()
    return tweet_data
