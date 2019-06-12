from typing import List

import torch
from flair.data import TaggedCorpus, Sentence, Token
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.training_utils import clear_embeddings
from sklearn import metrics


def _evaluate_sequence_tagger(model,
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

            tags, loss = model.forward_labels_and_loss(batch)
            eval_loss += loss

            for (sentence, sent_tags) in zip(batch, tags):
                for (token, tag) in zip(sentence.tokens, sent_tags):
                    token: Token = token
                    token.add_tag_label('predicted', tag)

                gold_tags = [tag.tag for tag in sentence.get_spans(model.tag_type)]
                predicted_tags = [tag.tag for tag in sentence.get_spans('predicted')]
                gold_seqs.append(gold_tags)
                pred_seqs.append(predicted_tags)

            clear_embeddings(batch, also_clear_word_embeddings=not embeddings_in_memory)

        eval_loss /= len(sentences)

        scores = calc_seqtag_eval_scores(gold_seqs, pred_seqs)
        return scores, eval_loss


def calc_seqtag_eval_scores(gold_seqs, pred_seqs):
    gold_flattened = [l for seq in gold_seqs for l in seq]
    pred_flattened = [l for seq in pred_seqs for l in seq]
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

    corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(NLPTask.UD_ENGLISH)
    print(corpus)

    tag_type = 'pos'

    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings('glove'),
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    from flair.models import SequenceTagger

    tagger = SequenceTagger = SequenceTagger.load_from_file(
        'sequence_tagging/resources/taggers/example-ner/final-model.pt')

    metric2, eval_loss = _evaluate_sequence_tagger(tagger, corpus.test)
    print(metric2)