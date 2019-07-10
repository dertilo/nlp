from pprint import pprint
from typing import List

import torch
from flair.data import Sentence, Corpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.training_utils import clear_embeddings
from sklearn import metrics
from flair.models import SequenceTagger

from sequence_tagging.seq_tag_util import spanwise_pr_re_f1, bilou2bio


def calc_train_test_spanwise_f1(tagger:SequenceTagger, train_sentences, test_sentences, tag_name='pos'):
    gold_targets_train = extract_gold_tags_from_flair_sent(tag_name, train_sentences)
    gold_targets_test = extract_gold_tags_from_flair_sent(tag_name, test_sentences)
    pred_data = flair_tagger_predict_bio(tagger, tag_name, train_sentences)
    _, _, f1_train = spanwise_pr_re_f1(pred_data, gold_targets_train)
    pred_data = flair_tagger_predict_bio(tagger, tag_name, test_sentences)
    _, _, f1_test = spanwise_pr_re_f1(pred_data, gold_targets_test)
    return f1_train,f1_test

def calc_print_f1_scores(tagger:SequenceTagger, train_sentences, test_sentences, tag_name='pos'):
    gold_targets_train = extract_gold_tags_from_flair_sent(tag_name, train_sentences)
    gold_targets_test = extract_gold_tags_from_flair_sent(tag_name, test_sentences)

    pred_data = flair_tagger_predict_bio(tagger, tag_name, train_sentences)
    pprint('train-f1-macro: %0.2f' % calc_seqtag_eval_scores(gold_targets_train, pred_data)['f1-macro'])
    pprint('train-f1-micro: %0.2f' % calc_seqtag_eval_scores(gold_targets_train, pred_data)['f1-micro'])
    _, _, f1 = spanwise_pr_re_f1(pred_data, gold_targets_train)
    pprint('train-f1-spanwise: %0.2f' % f1)

    pred_data = flair_tagger_predict_bio(tagger, tag_name, test_sentences)
    pprint('test-f1-macro: %0.2f' % calc_seqtag_eval_scores(gold_targets_test, pred_data)['f1-macro'])
    pprint('test-f1-micro: %0.2f' % calc_seqtag_eval_scores(gold_targets_test, pred_data)['f1-micro'])
    _, _, f1 = spanwise_pr_re_f1(pred_data, gold_targets_test)
    pprint('train-f1-spanwise: %0.2f' % f1)


def extract_gold_tags_from_flair_sent(tag_name:str, sent_flair:Sentence):
    train_data = [[(token.text, token.tags[tag_name].value) for token in datum] for datum in sent_flair]
    gold_targets_train = [bilou2bio([tag for token, tag in datum]) for datum in train_data]
    return gold_targets_train


def flair_tagger_predict_bio(tagger, tag_name, train_sentences):
    pred_sentences = tagger.predict(train_sentences)
    pred_data = [bilou2bio([token.tags[tag_name].value for token in datum]) for datum in pred_sentences]
    return pred_data


def evaluate_sequence_tagger(model:SequenceTagger,
                             sentences: List[Sentence],
                             eval_mini_batch_size: int = 32,
                             embeddings_in_memory: bool = True,
                             ) -> (dict, float):
    with torch.no_grad():
        eval_loss = 0

        batch_no: int = 0
        batches = [sentences[x:x + eval_mini_batch_size] for x in range(0, len(sentences), eval_mini_batch_size)]

        gold_seqs = []
        pred_seqs = []

        for batch in batches:
            batch_no += 1

            features = model.forward(batch)
            loss = model._calculate_loss(features, batch)
            pred_tags = model._obtain_labels(features, batch)

            eval_loss += loss

            for (sentence, pred_sent_tags) in zip(batch, pred_tags):
                gold_tags = [tok.tags['ner'].value for tok in sentence]
                predicted_tags = [l.value for l in pred_sent_tags]
                gold_seqs.append(gold_tags)
                pred_seqs.append(predicted_tags)

            clear_embeddings(batch, also_clear_word_embeddings=not embeddings_in_memory)

        eval_loss /= len(sentences)

        scores = calc_seqtag_eval_scores(gold_seqs, pred_seqs)
        scores['eval-loss']=eval_loss
        return scores


def calc_seqtag_eval_scores(gold_seqs, pred_seqs):
    gold_flattened = [l for seq in gold_seqs for l in seq]
    pred_flattened = [l for seq in pred_seqs for l in seq]
    assert len(gold_flattened) == len(pred_flattened)
    labels = list(set(gold_flattened))
    scores = {
        'f1-micro': metrics.f1_score(gold_flattened, pred_flattened, average='micro'),
        'f1-macro': metrics.f1_score(gold_flattened, pred_flattened, average='macro'),
        'clf-report': metrics.classification_report(gold_flattened, pred_flattened, target_names=labels, digits=3,
                                                    output_dict=True),
    }
    return scores



if __name__ == '__main__':
    from pathlib import Path
    home = str(Path.home())
    data_path = 'data/scierc_data/json/'
    # corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(NLPTask.UD_ENGLISH)
    from sequence_tagging.flair_scierc_ner import read_scierc_data_to_FlairSentences
    corpus = Corpus(
        train=read_scierc_data_to_FlairSentences('%strain.json' % data_path),
        dev=read_scierc_data_to_FlairSentences('%sdev.json' % data_path),
        test=read_scierc_data_to_FlairSentences('%stest.json' % data_path), name='scierc')

    print(corpus)

    tag_type = 'pos'

    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings('glove'),
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)


    tagger = SequenceTagger = SequenceTagger.load(
        'sequence_tagging/resources/taggers/scierc-ner/final-model.pt')

    pprint(evaluate_sequence_tagger(tagger, corpus.train))
    pprint(evaluate_sequence_tagger(tagger, corpus.test))