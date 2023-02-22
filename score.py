import numpy as np
from sklearn.metrics import make_scorer, matthews_corrcoef


def processed_score(score, yval, ypred, bottom=-0.005, top=0.0055, thresh=0.0):
    # scores as processed in paper
    if thresh is None:
        thresh = np.mean([bottom, top])
    inbetw = (yval > bottom) & (yval < top)
    yval = (yval > thresh)[~inbetw]
    ypred = (ypred > thresh)[~inbetw]
    return score(yval, ypred)


def matthews(y_true, y_pred):
    return processed_score(matthews_corrcoef, y_true, y_pred, bottom=0, top=0)


scorer = make_scorer(matthews, greater_is_better=True)
