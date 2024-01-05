import numpy as np 
from itertools import combinations

def enrich_df_with_features(df):

    prices = ['reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap']
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]

    df_ = df.copy()

    df_['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)')
    df_['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)')  

    for c in combinations(prices, 2):
        df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

    for a, b, c in combinations( ['reference_price', 'ask_price', 'bid_price', 'wap'], 3):
        maxi = df_[[a,b,c]].max(axis=1)
        mini = df_[[a,b,c]].min(axis=1)
        mid = df_[[a,b,c]].sum(axis=1)-mini-maxi

        df_[f'{a}_{b}_{c}_imb2'] = np.where(mid.eq(mini), np.nan, (maxi - mid) / (mid - mini))

    def calculate_pressure(df):
        return np.where(
            df['imbalance_buy_sell_flag']==1,
            df['imbalance_size']/df['ask_size'],
            df['imbalance_size']/df['bid_size']
        )
    
    df_['pressure'] = calculate_pressure(df)
    df_['inefficiency'] = df.eval('imbalance_size/matched_size')

    return df_