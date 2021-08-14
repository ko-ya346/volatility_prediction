import numpy as np


def calc_wap(df):
    """
    wap(加重平均価格)を計算する
    """
    wap = (df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]) / (
        df["bid_size1"] + df["ask_size1"]
    )
    return wap


def calc_wap2(df):
    """
    wap(加重平均価格)を計算する
    """
    wap = (df["bid_price2"] * df["ask_size2"] + df["ask_price2"] * df["bid_size2"]) / (
        df["bid_size2"] + df["ask_size2"]
    )
    return wap


def log_return(stock_prices):
    """
    log returnを計算する。

    Parameter
    ---------
    stock_prices: list[float]
        価格のリスト
    """
    return np.log(stock_prices).diff()


def realized_volatility(log_returns):
    """
    volatilityを計算する。

    Parameter
    ---------
    log_returns: list[float]
        log returnの配列
    """
    return np.sqrt(np.sum(log_returns ** 2))


def count_unique(series):
    return len(np.unique(series))


def rmspe(y_true, y_pred):
    return (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))


def feval_RMSPE(preds, lgbm_train):
    """
    lightgbmの評価用
    """
    labels = lgbm_train.get_label()
    return 'RMSPE', round(rmspe(y_true = labels, y_pred = preds),5), False
