# import libraries


import pandas as pd
import numpy as np
import math
import json
import datetime
from sklearn import preprocessing
from collections import Counter


def load_data():
    '''
    This funtion is to load the data from database
    
    Argument:

    None

    return:

    
    portfolio
    profile
    transcript
    '''
    portfolio = pd.read_json('portfolio.json', orient='records', lines=True)
    profile = pd.read_json('profile.json', orient='records', lines=True)
    transcript = pd.read_json('transcript.json', orient='records', lines=True)
    
    return portfolio,profile,transcript


def feature_engineer(portfolio,profile,transcript):
    '''
    Clean up the dataset and featrue engineering on different dataset
    '''

    ## clean up transcript dataset and to feature engineer, get value fro json data type
    transcript['offer_id'] = transcript['value'].map(lambda x: x['offer id'] if 'offer id' in x.keys() else (x['offer_id'] if 'offer_id' in x.keys() else np.nan))
    transcript['amount'] =  transcript['value'].map(lambda x: x['amount'] if 'amount' in x.keys() else np.nan)
    transcript['reward'] = transcript['value'].map(lambda x: x['reward'] if 'reward' in x.keys() else np.nan)
    
    ## merge three dataset together
    transcript_portfolio = pd.merge(transcript, portfolio, how='left', left_on=['offer_id'], right_on=['id'])
    transcript_merge = pd.merge(transcript_portfolio, profile, how='left', left_on=['person'], right_on=['id'])
    transcript_merge.drop(['id_x','id_y'],axis=1,inplace=True)

    ## Do some statistic for transaction history as new data
    profile_info = {}
    portfolio_info = {}
    for n,g in transcript_portfolio.groupby('person'):
        event_dict = {}
        event_dict['transaction'] = len(g[g['event']=='transaction'])
        event_dict['avg_transaction'] = g[g['event']=='transaction']['amount'].mean()
        event_dict['avg_transaction_high'] = g[g['event']=='transaction']['amount'].max()
        event_dict['avg_transaction_low'] = g[g['event']=='transaction']['amount'].min()
        event_dict['avg_transaction_std'] = g[g['event']=='transaction']['amount'].std()
        event_dict['transaction_time_var'] = np.log1p(g[g['event']=='transaction']['time'].var())
        event_dict['informational_num'] = len(g[g['offer_type']=='informational'])
        profile_info[n] = event_dict


    profile_portfolio_dict = {}
    for n,g in transcript_merge.groupby(['person','offer_id']):
        event_dict = dict(Counter(g['event']))
        profile_portfolio_dict[n] = event_dict


    profile_w_records = profile.set_index('id').merge(pd.DataFrame.from_dict(profile_info,orient='index'),left_index=True, right_index=True)
    portfolio_w_records = portfolio
    profile_w_records.index.names=['id']

    ## Labelize non-digit features
    le = dict()
    le['gender'] = preprocessing.LabelEncoder()
    le['offer_type'] = preprocessing.LabelEncoder()
    le['channels_'] = preprocessing.LabelEncoder()
    
    ## Re-engineering the existing features for later study
    portfolio_w_records['from_web'] = portfolio_w_records['channels'].map(lambda x: 1 if 'web' in x else 0)
    portfolio_w_records['from_email'] = portfolio_w_records['channels'].map(lambda x: 1 if 'email' in x else 0)
    portfolio_w_records['from_mobile'] = portfolio_w_records['channels'].map(lambda x: 1 if 'mobile' in x else 0)
    portfolio_w_records['from_social'] = portfolio_w_records['channels'].map(lambda x: 1 if 'social' in x else 0)
    portfolio_w_records['channels_'] = portfolio_w_records['channels'].map(lambda x: ','.join(x))
    portfolio_w_records['offer_type'] = le['offer_type'].fit_transform(portfolio_w_records['offer_type'])
    portfolio_w_records['channels_'] = le['channels_'].fit_transform(portfolio_w_records['channels_'])

    ## Create some time related features for study
    profile_w_records = profile_w_records.dropna(subset=['income'])
    profile_w_records['became_member_on']=pd.to_datetime(profile_w_records['became_member_on'],format='%Y%m%d')
    profile_w_records.sort_values('became_member_on',ascending=False)
    profile_w_records['became_member_year'] = profile_w_records['became_member_on'].map(lambda x:x.year)
    profile_w_records['became_member_month'] = profile_w_records['became_member_on'].map(lambda x:x.month)
    profile_w_records['became_member_weekday'] = profile_w_records['became_member_on'].dt.dayofweek
    profile_w_records['today'] = pd.to_datetime(datetime.datetime.today().date())
    profile_w_records['join_days'] = (profile_w_records['today']-profile_w_records['became_member_on']).dt.days
    profile_w_records['join_years'] = (profile_w_records['join_days']/365).astype('int')
    profile_w_records['join_age'] = (profile_w_records['age']-profile_w_records['join_years'])
    profile_w_records['gender'] = le['gender'].fit_transform(profile_w_records['gender'])

    profile_portfolio_df = pd.DataFrame.from_dict(profile_portfolio_dict,orient='index')
    profile_portfolio_df.index.names = ['person_id','portfolio_id']
    profile_portfolio_df = profile_portfolio_df.reset_index()
    profile_portfolio_df = profile_portfolio_df.fillna(0)
    profile_w_records = profile_w_records.reset_index()
    profile_portfolio_df = pd.merge(profile_portfolio_df,profile_w_records,how='left',left_on='person_id',right_on='id')
    profile_portfolio_df = pd.merge(profile_portfolio_df,portfolio_w_records,how='left',left_on='portfolio_id',right_on='id')
    profile_portfolio_df['complete_rate'] = profile_portfolio_df['offer completed']/profile_portfolio_df['offer received']


    ## Extract useful features for study
    profile_portfolio_df_ = profile_portfolio_df[['age', 'gender', 'income',
       'join_days', 'join_years', 'join_age',
       'difficulty', 'duration', 'offer_type', 'reward', 'from_web',
       'from_email', 'from_mobile', 'from_social', 'channels_','informational_num','transaction','transaction_time_var','avg_transaction','avg_transaction_high',
       'complete_rate']]

    ## Only keep complete rate 1 or 0 for simple classification learning
    profile_portfolio_df_ = profile_portfolio_df_[(profile_portfolio_df_.complete_rate==1) | (profile_portfolio_df_.complete_rate==0)]

    return profile_portfolio_df_
    
def save_data(df, name):

    '''
    Save clean data to database

    Arg:
    df: clean dataframe
    name: csv file_path


    '''
    df.to_csv(name)  


def main():

    '''
    Data Processing procedure.
    
    '''

    print('Loading data...\n')
    portfolio,profile,transcript = load_data()

    print('Featureing data...')
    df = feature_engineer(portfolio,profile,transcript)
    
    print('Saving data...\n')
    save_data(df,'data.csv')
    
    print('Cleaned data saved!')


if __name__ == '__main__':
    main()