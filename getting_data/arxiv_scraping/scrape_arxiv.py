import gzip
import json
from typing import Iterable

from arxivscraper import Scraper


def write_jsons_to_file(file:str,data:Iterable, mode="wb"):
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
        f.writelines((process_line(d) for d in data))

if __name__ == '__main__':
    scraper = Scraper(category='stat',
                         date_from='2019-01-01',
                         date_until='2019-01-31',
                         retry_delay_seconds=10,
                         filters={'categories': ['stat.ml'], 'abstract': ['learning']})

    write_jsons_to_file('./scraped.jsonl',scraper.scrape(100))