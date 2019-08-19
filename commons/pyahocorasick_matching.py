import ahocorasick
from time import time
from typing import Iterable, List

from commons import data_io


class PyAhoCorasickMatcher(object):
    def __init__(self, phrases_leaves_g:Iterable):
        start = time()
        self.A = ahocorasick.Automaton()
        [self.A.add_word(phrase, leaf) for phrase,leaf in phrases_leaves_g]
        self.A.make_automaton()
        print('took %0.2f seconds to stuff %d phrases in trie' % (time()-start,len(self.A)))

    def get_matches(self,s:str)->List:
        return [val for end_index, val in self.A.iter(s)]

# def build_tags_matcher(tags_repo_path,tag_filter_fun=lambda x:True)->PyAhoCorasickMatcher:
#     g = ((tag,phrases) for tag,phrases in build_tag2phrases_lookup(tags_repo_path=tags_repo_path).items())
#
#     def cut_leaf(tag):
#         return '/'.join(tag.split('/')[:-1])
#
#     phrases_leaves = [(phrase.lower(), cut_leaf(tag)) for tag,phrases in g if tag_filter_fun(tag) for phrase in phrases ]
#     m = PyAhoCorasickMatcher(phrases_leaves_g=phrases_leaves)
#     return m

# def build_searchphrase_matcher(tags_repo_path)->PyAhoCorasickMatcher:
#     phrase2id = {phrase:eid for tag,phrases_ids in build_path2phraseids_lookup(tags_repo_path=tags_repo_path).items() for phrase,eid in phrases_ids}
#     phrases_leaves = [(phrase.lower(), phrase.lower().replace(' ','_')) for phrase,eid in phrase2id.items()]
#     m = PyAhoCorasickMatcher(phrases_leaves_g=phrases_leaves)
#     return m
#
# def build_searchphrase_matcher_form_searchphrase2id_file(searchphrase2id_file:str)->PyAhoCorasickMatcher:
#     phrase2id = ((p,i) for p,i in data_io.read_jsons_from_file(searchphrase2id_file))
#     phrases_leaves = [(phrase.lower(), phrase.lower().replace(' ','_')) for phrase,eid in phrase2id]
#     m = PyAhoCorasickMatcher(phrases_leaves_g=phrases_leaves)
#     return m