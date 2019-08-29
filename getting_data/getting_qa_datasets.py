import json

import parlai.core.build_data as build_data
import os

from commons import data_io
from parlai.tasks.wikiqa.build import create_fb_format


def download_process_wikiqa(
        data_path = 'data'
    ):
    dpath = os.path.join(data_path, 'WikiQA')
    build_data.make_dir(dpath)
    fname = 'wikiqa.tar.gz'
    url = 'http://parl.ai/downloads/wikiqa/' + fname
    build_data.download(url, dpath, fname)
    build_data.untar(dpath, fname)
    dpext = os.path.join(dpath, 'WikiQACorpus')
    create_fb_format(dpath, 'train', os.path.join(dpext, 'WikiQA-train.tsv'))
    create_fb_format(dpath, 'valid', os.path.join(dpext, 'WikiQA-dev.tsv'))
    create_fb_format(dpath, 'test', os.path.join(dpext, 'WikiQA-test.tsv'))
    create_fb_format(
        dpath, 'train-filtered', os.path.join(dpext, 'WikiQA-train.tsv')
    )
    create_fb_format(dpath, 'valid-filtered', os.path.join(dpext, 'WikiQA-dev.tsv'))
    create_fb_format(dpath, 'test-filtered', os.path.join(dpext, 'WikiQA-test.tsv'))

def download_natural_questions(
        data_path = 'data'
    ):
    data_path = '/home/tilo/Downloads/v1.0_sample_nq-train-sample.jsonl.gz'
    data = list(data_io.read_jsons_from_file(data_path))
    print()


def load_qangaroo_dataset(
        data_path, dataset='wikihop',split_name='train'
    ):
    def download_if_not_existing(data_path):
        dpath = os.path.join(data_path, 'qangaroo')
        if not os.path.isdir(dpath):
            os.mkdir(dpath)
            fname = 'qangaroo.zip'
            g_ID = "1ytVZ4AhubFDOEL7o7XrIRIyhU8g9wvKA"

            print("downloading ...")
            build_data.download_from_google_drive(g_ID, os.path.join(dpath, fname))
            build_data.untar(dpath, fname)

    download_if_not_existing(data_path)

    file_str = '%s/qangaroo/qangaroo_v1.1/%s/%s.json'%(data_path,dataset,split_name)
    with open(file_str) as data_file:
        data = json.load(data_file)
    return data

def download_coqa(data_dir):
    data_dir = data_dir+'/coqa'
    file = 'coqa-train-v1.0.json'
    dev_file = 'coqa-dev-v1.0.json'
    base_url = 'https://nlp.stanford.edu/data/coqa'
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
        os.system('wget -N -q -P %s %s' % (data_dir, '%s/%s' % (base_url, file)))
        os.system('wget -N -q -P %s %s' % (data_dir, '%s/%s' % (base_url, dev_file)))

if __name__ == '__main__':
    # download_process_wikiqa()
    # download_natural_questions()
    download_coqa('/home/tilo/data/QA')

