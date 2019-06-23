from pprint import pprint
from typing import List

import torch
from flair.data import Sentence, Corpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.training_utils import clear_embeddings
from sklearn import metrics
from flair.models import SequenceTagger


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

            with torch.no_grad():
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