
'''
do
wget cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
unzip trainingandtestdata.zip

The data is a CSV with emoticons removed. Data file format has 6 fields:
0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
1 - the id of the tweet (2087)
2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
3 - the query (lyx). If there is no query, then this value is NO_QUERY.
4 - the user that tweeted (robotickilldozr)
5 - the text of the tweet (Lyx is cool)

(see http://help.sentiment140.com/for-students/)

'''
import gzip
import sys

def parse_line(line:str):
    s = line.split('","')
    polarity,tweet_id,date,query,user,text = s

    return {
        'polarity':int(polarity.replace('"','')),
        'tweet_id':tweet_id,
        'date':date,
        'query':query,
        'user':user,
        'text':text[:-1] # very last is a "
    }


def read_lines(file, mode ='b', encoding ='utf-8', limit=sys.maxsize):
    counter = 0
    with gzip.open(file, mode='r'+mode) if file.endswith('.gz') else open(file,mode='r'+mode) as f:
        for line in f:
            counter+=1
            if counter>limit:
                break
            if mode == 'b':
                try:
                    yield line.decode(encoding).replace('\n', '')
                except:
                    pass
            elif mode == 't':
                yield line.replace('\n','')


def get_sentiment140_data(datafile,limit=sys.maxsize):
    return (parse_line(line) for line in read_lines(datafile,limit=limit))