#%%

from psaw import PushshiftAPI
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm
import datetime
import os
from itertools import islice
from collections import defaultdict
import networkx as nx
import warnings
if not os.path.exists('./data/'):
    raise FileNotFoundError("Data foulder wasn't found.\nMake sure it exists, and contains the right files before proceeing")
warnings.filterwarnings("ignore")


def load_one_day_subs(start_time: datetime.datetime, subreddits, query, fields, api):
    """
    the name speaks for itself
    """
    after = int(start_time.timestamp())
    before = int((start_time + datetime.timedelta(days = 1)).timestamp())
    submission_generator = api.search_submissions(subreddit=','.join(subreddits),
                                                    after=after,
                                                    before=before,
                                                    q=query,
                                                    fields=fields)
    sumbissions = pd.DataFrame([obj.d_ for obj in submission_generator])

    return sumbissions
#%%
start_time = datetime.datetime(2022,3,28) # Oscars started midnight UTC
end_time = datetime.datetime(2022,4,18)
n_total_days = (end_time - start_time).days

subreddits = ['news', 'television', 'worldnews', 'USnews','qualitynews', 'offbeat', 'OutOfTheLoop', 'Oscars', 'boxoffice', 'willsmith', 'entertainment' ]

query = 'slap|will|smith|chris|rock|oscars|oscar'

fields = ['author', 'author_fullname', 'created_utc', 'id', 'num_comments',
          'score', 'subreddit', 'subreddit_id', 'title', 'upvote_ratio', 'created']

api = PushshiftAPI() 

#%%

if 'submissions.pkl' in os.listdir('data'):
    data_submissions = pd.read_pickle('data/submissions.pkl')
else:
    results = [load_one_day_subs(start_time + datetime.timedelta(days = i),
                        subreddits,
                        query,
                        fields,
                        n_total_days,
                        api) for i in tqdm(range(n_total_days))]

    data_submissions = pd.concat(results, ignore_index = True)
    data_submissions.to_pickle('data/submissions.pkl')

#%%
comment_fields = ['author', 'body', 'controversiality', 'created_utc',
                  'id', 'link_id', 'parent_id', 'score', 'score_hidden', 'subreddit', 'subreddit_id']

if 'comments.pkl' in os.listdir('data'):
    data_comments = pd.read_pickle('data/comments.pkl')

else:
    n,m = data_submissions.shape
    # n = 100
    data_comments = [None]*n
    for i in tqdm(range(n)):
        subreddit, id = data_submissions[['subreddit', 'id']].iloc[i]
        
        comments_generator = api.search_comments(subreddit=subreddit,
                                                link_id=id,
                                                fields=comment_fields)
        
        data_comments[i] = pd.DataFrame([obj.d_ for obj in comments_generator])
    data_comments = pd.concat(data_comments, ignore_index=True)
    data_comments.to_pickle('data/comments.pkl')

print(data_comments)

#%%

data_comments

