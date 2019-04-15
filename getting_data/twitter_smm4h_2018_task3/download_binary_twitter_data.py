#!/usr/bin/python
import gzip
import os
import sys, urllib, re, json, socket, string
from bs4 import BeautifulSoup
socket.setdefaulttimeout(20)


def read_jsons_from_file(file, limit=sys.maxint):
    with gzip.open(file, mode="rb") if file.endswith('.gz') else open(file, mode="rb") as f:
        counter=0
        for line in f:
            # assert isinstance(line,bytes)
            counter += 1
            if counter > limit: break
            yield json.loads(line.decode('utf-8'))


def read_lines_from_txt_files(path, mode ='b', encoding ='utf-8', limit=sys.maxint):
    g = (line for file in os.listdir(path) if file.endswith('.txt')
         for line in read_lines(path + '/' + file,mode,encoding))
    c=0
    for line in g:
        c+=1
        if c>limit:break
        yield line

def read_lines(file, mode ='b', encoding ='utf-8', limit=sys.maxint):
    counter = 0
    with gzip.open(file, mode='r'+mode) if file.endswith('.gz') else open(file,mode='r'+mode) as f:
        for line in f:
            counter+=1
            if counter>limit:
                break
            if mode == 'b':
                yield line.decode(encoding).replace('\n','')
            elif mode == 't':
                yield line.replace('\n','')

def run_twitter_scaping(dir,limit=sys.maxint):
    c = 0
    for line in read_lines_from_txt_files(dir):
        fields = line.rstrip().split('\t')
        tweetid = fields[0]
        userid = fields[1]

        if tweetid in twitter_ids:
            continue
        else:
            twitter_ids.add(tweetid)

        if len(fields)==4:
            label = fields[3]
        elif len(fields)==3:
            label = fields[2]
        else:
            label = None
            print(line)

        f = urllib.urlopen('http://twitter.com/' + str(userid) + '/status/' + str(tweetid))
        html = f.read().replace("</html>", "") + "</html>"
        soup = BeautifulSoup(html)
        jstt = soup.find_all("p", "js-tweet-text")
        if len(jstt)>0:
            text = get_text(jstt, soup)
        else:
            text = None
        c+=1
        sys.stdout.write('\r%d'%c)
        sys.stdout.flush()
        yield {
           'tweetid':tweetid,
            'userid':userid,
            'text':text,
            'label':label,
        }
        if c>limit:
            break


def get_text(jstt, soup):
    tweets = list(set([x.get_text() for x in jstt]))
    if (len(tweets)) > 1:
        other_tweets = []
        cont = soup.find_all("div", "content")
        for i in cont:
            o_t = i.find_all("p", "js-tweet-text")
            other_text = list(set([x.get_text() for x in o_t]))
            other_tweets += other_text
        tweets = list(set(tweets) - set(other_tweets))
    if len(tweets)>0:
        text = tweets[0]

        for j in soup.find_all("input", "json-data", id="init-data"):
            js = json.loads(j['value'])
            if (js.has_key("embedData")):
                tweet = js["embedData"]["status"]
                text = js["embedData"]["status"]["text"]
                break
        text = text.replace('\n', ' ', )
        text = re.sub(r'\s+', ' ', text)
    else:
        text = None
    return text


def write_jsons_to_file(file,data, mode="wb"):
    def process_line(hit):
        try:
            line = json.dumps(hit,skipkeys=True, ensure_ascii=False)
            line = line + "\n"
            line = line.encode('utf-8')
        except Exception as e:
            line=''
            print(e)
        return line
    with gzip.open(file, mode=mode) if file.endswith('gz') else open(file, mode=mode) as f:
        for d in data:
            f.write(process_line(d))
            f.flush()
        # f.writelines((process_line(d) for d in data))


if __name__ == '__main__':
    f = './tweets.jsonl'
    if os.path.isfile(f):
        twitter_ids = set([d['tweetid'] for d in read_jsons_from_file(f)])
    else:
        twitter_ids = set()
    print('already got %d tweets'%len(twitter_ids))

    g = run_twitter_scaping('.')
    write_jsons_to_file(f,g,mode='ab')

