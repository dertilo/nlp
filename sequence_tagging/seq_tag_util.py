from typing import Tuple, List

import numpy as np
from sklearn import metrics


def calc_seqtag_f1_scores(pred_targets_fun, token_tag_sequences:List[List[Tuple[str, str]]]):
    assert len(token_tag_sequences)>0
    y_pred,targets = pred_targets_fun(token_tag_sequences)
    _,_,f1_train = spanwise_pr_re_f1(y_pred, targets)
    return {
        'f1-macro':calc_seqtag_tokenwise_scores(targets, y_pred)['f1-macro'],
        'f1-micro':calc_seqtag_tokenwise_scores(targets, y_pred)['f1-micro'],
        'f1-spanwise':f1_train
    }

def calc_seqtag_tokenwise_scores(gold_seqs, pred_seqs):
    gold_flattened = [l for seq in gold_seqs for l in seq]
    pred_flattened = [l for seq in pred_seqs for l in seq]
    assert len(gold_flattened) == len(pred_flattened) and len(gold_flattened)>0
    labels = list(set(gold_flattened))
    scores = {
        'f1-micro': metrics.f1_score(gold_flattened, pred_flattened, average='micro'),
        'f1-macro': metrics.f1_score(gold_flattened, pred_flattened, average='macro'),
        'clf-report': metrics.classification_report(gold_flattened, pred_flattened, target_names=labels, digits=3,
                                                    output_dict=True),
    }
    return scores

def mark_text(text, char_spans):
    sorted_spans = sorted(char_spans, key=lambda sp:-sp[0])
    for span in sorted_spans:
        assert span[1]>span[0]
        text = text[:span[1]]+'</'+span[2]+'>'+text[span[1]:]
        text = text[:span[0]]+'<'+span[2]+'>'+text[span[0]:]
    return text

def correct_biotags(tag_seq):
    correction_counter = 0
    corr_tag_seq = tag_seq
    for i in range(len(tag_seq)):
        if i>0 and tag_seq[i-1] is not 'O':
            previous_label = tag_seq[i-1][2:]
        else:
            previous_label = 'O'
        current_label = tag_seq[i][2:]
        if tag_seq[i].startswith('I-') and not current_label is not previous_label:
            correction_counter+=1
            corr_tag_seq[i]='B-'+current_label
    return corr_tag_seq

def bilou2bio(tag_seq):
    '''
    BILOU to BIO
    or
    BIOES to BIO
    E == L
    S == U
    '''
    bio_tags = tag_seq
    for i in range(len(tag_seq)):
        if tag_seq[i].startswith('U-') or tag_seq[i].startswith('S-'):
            bio_tags[i] = 'B-'+tag_seq[i][2:]
        elif tag_seq[i].startswith('L-') or tag_seq[i].startswith('E-'):
            bio_tags[i] = 'I-'+tag_seq[i][2:]
    return bio_tags
#   public static List<Triple<Integer, Integer, String>> tokenTagsToSpans(
#           List<Pair<Integer,Integer>> startEndOffsets, List<String> tags) {
#     List<Triple<Integer, Integer, String>> targetSpans = new ArrayList<>();
#     for (int i = 0; i < startEndOffsets.size(); i++) {
#       if (tags.get(i).startsWith("B")) {
#         String label = tags.get(i).substring(2);
#         int startIndex = startEndOffsets.get(i).getFirst();
#         i++;
#         while (i < tags.size()
#             && (tags.get(i).startsWith("I") || tags.get(i).startsWith("L"))
#             && tags.get(i).substring(2).equals(label)) {
#           i++;
#         }
#         i--;
#         int endIndex = startEndOffsets.get(i).getSecond();
#         targetSpans.add(new Triple<>(startIndex, endIndex, label));
#       }
#     }
#     return targetSpans;
#   }
def bio_to_token_spans(tag_seq):
    spans = []
    for i in range(len(tag_seq)):
        if tag_seq[i].startswith('B-'):
            label = tag_seq[i][2:]
            startIdx = i
            i+=1
            while i<len(tag_seq) and tag_seq[i].startswith('I-') and tag_seq[i][2:] == label:
                i+=1
            i-=1
            spans.append((startIdx,i,label))
    return spans

def char_precise_spans_to_token_spans(char_precise_spans,start_ends):
    if all([isinstance(span,dict) for span in char_precise_spans]):
        char_precise_spans = [(s['startIdx'],s['endIdx'],s['label']) for s in char_precise_spans]
    spans = []
    for span in char_precise_spans:
        closest_token_start = int(np.argmin([np.abs(x[0] - span[0]) for x in start_ends]))
        closest_token_end = int(np.argmin([np.abs(x[1] - span[1]) for x in start_ends]))
        spans.append((closest_token_start,closest_token_end,span[2]))
    return spans

def char_precise_spans_to_BIO_tagseq(char_precise_spans:List[Tuple[int, int, str]], start_ends:List[Tuple[int,int]]):
    tags = ['O' for _ in range(len(start_ends))]
    for span in char_precise_spans:
        closest_token_start = np.argmin([np.abs(x[0] - span[0]) for x in start_ends])
        closest_token_end = np.argmin([np.abs(x[1] - span[1]) for x in start_ends])
        if closest_token_end - closest_token_start == 0:
            tags[closest_token_start] = 'B-' + span[2]  # 'U-'+span[2]
        else:
            tags[closest_token_start] = 'B-' + span[2]
            tags[closest_token_end] = 'I-' + span[2]  # 'L-'+span[2]
            for id in range(closest_token_start + 1, closest_token_end):
                tags[id] = 'I-' + span[2]
    # assert all([tag in tagSet for tag in tags])
    return [tag for tag in tags]

def tags_to_token_spans(tag_seq):
    return bio_to_token_spans(bilou2bio(tag_seq))

#TODO: might be deprecated!
def probas_to_tagseq(pred_probas,lenghtes,target_binarizer)->Tuple[List[List[str]],List[List[float]]]:
    assert isinstance(lenghtes, list) and all([isinstance(x, int) for x in lenghtes])
    assert isinstance(pred_probas,np.ndarray)
    sequences = []; proba_sequences=[]
    for pred_proba, lenghts in zip(pred_probas, lenghtes):
        pred_proba = np.squeeze(pred_proba)
        binarized_prediction = np.array([1.0 * (probas == np.amax(probas)) for probas in pred_proba.tolist()])
        tags = target_binarizer.inverse_transform(binarized_prediction)
        proba_sequence = [np.amax(probas) for probas in pred_proba.tolist()]
        tags_ = [t[0] if len(t) > 0 else 'O' for t in tags]
        if len(tags_) > lenghts:
            tagseq = tags_[:lenghts]
            proba_sequence = proba_sequence[:lenghts]
        else:
            tagseq = tags_ + ['O' for _ in range(lenghts - len(tags_))]
            proba_sequence = proba_sequence+[0.5 for _ in range(lenghts - len(tags_))]
        proba_sequences.append(proba_sequence)
        sequences.append(tagseq)
    assert all([len(seq)==len(prob) for seq,prob in zip(sequences,proba_sequences)])
    return sequences,proba_sequences

def probas_list_to_tagseq(pred_probas:List[np.ndarray],lenghtes,target_binarizer)->Tuple[List[List[str]],List[List[float]]]:
    assert isinstance(lenghtes, list) and all([isinstance(x, int) for x in lenghtes])
    assert isinstance(pred_probas,list) and all([isinstance(x,np.ndarray) for x in pred_probas])
    sequences = [];
    max_proba_sequences=[] # might be useful for activelearning
    for pred_proba, lenghts in zip(pred_probas, lenghtes):
        binarized_prediction = np.array([1.0 * (probas == np.amax(probas)) for probas in pred_proba.tolist()])
        tags = target_binarizer.inverse_transform(binarized_prediction)
        proba_sequence = [np.amax(probas) for probas in pred_proba.tolist()]
        tags_ = [t[0] if len(t) > 0 else 'O' for t in tags]
        if len(tags_) > lenghts:
            tagseq = tags_[:lenghts]
            proba_sequence = proba_sequence[:lenghts]
        else:
            tagseq = tags_ + ['O' for _ in range(lenghts - len(tags_))]
            proba_sequence = proba_sequence+[0.5 for _ in range(lenghts - len(tags_))]
        max_proba_sequences.append(proba_sequence)
        sequences.append(tagseq)
    assert all([len(seq)==len(prob) for seq,prob in zip(sequences,max_proba_sequences)])
    return sequences,max_proba_sequences

def probas_to_char_spans(pred_probas_batch,start_endss,target_binarizer):
    assert all([isinstance(se,tuple) for se_seq in start_endss for se in se_seq])
    assert isinstance(pred_probas_batch, np.ndarray) and len(pred_probas_batch.shape) == 3 and pred_probas_batch.shape[0] == len(start_endss)
    token_spans_list = probas_to_token_spans(pred_probas_batch, [len(se) for se in start_endss], target_binarizer)
    return [token_spans_to_char_precise_spans(token_spans, start_ends) for token_spans,start_ends in zip(token_spans_list,start_endss)]


def probas_to_token_spans(pred_probas_batch, sequence_lenghtes, target_binarizer)->List[List[Tuple[int,int,str]]]:
    assert isinstance(pred_probas_batch, np.ndarray) and len(pred_probas_batch.shape) == 3
    token_spans_list = [tags_to_token_spans(tags) for tags in
                        probas_to_tagseq(pred_probas_batch, sequence_lenghtes,target_binarizer)[0]]
    return token_spans_list

def token_spans_to_char_precise_spans(token_spans:List[Tuple], start_ends:List[Tuple]):
    return [(start_ends[span[0]][0],start_ends[span[1]][1],span[2])
                      for span in token_spans]


def spanwise_pr_re_f1(label_pred, label_correct):
    pred_counts = [compute_TP_P(pred, gold) for pred,gold in zip(label_pred,label_correct)]
    gold_counts = [compute_TP_P(gold, pred) for pred,gold in zip(label_pred,label_correct)]
    prec = np.sum([x[0] for x in pred_counts]) / np.sum([x[1] for x in pred_counts])
    rec = np.sum([x[0] for x in gold_counts]) / np.sum([x[1] for x in gold_counts])
    f1 = 0
    if (rec + prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)
    return prec, rec, f1

def prefixed_tag_to_label(tag):
    return tag if tag=='O' else tag[2:]

def keep_label_of_interest(x,label):
    return "O" if prefixed_tag_to_label(x) != label else x

def calc_print_prefixed_seqtag_metrics(gold_seqs, pred_seqs, labels):
    '''
    prefixed with BIO or BILOU or whatever single-char prefix
    '''

    assert all([len(p)==len(g) for p,g in zip(pred_seqs,gold_seqs)])
    f1_scores ={}; prec={}; rec= {}
    all_labels="all"

    for label in labels:
        pred_single_label = [[keep_label_of_interest(x,label) for x in sequence] for sequence in pred_seqs]
        gold_single_label = [[keep_label_of_interest(x,label) for x in sequence] for sequence in gold_seqs]
        prec[label], rec[label], f1_scores[label] = spanwise_pr_re_f1(pred_single_label, gold_single_label)
        print(label+": Prec: %.3f, Rec: %.3f, F1: %.3f" % (prec[label], rec[label], f1_scores[label]))
    prec[all_labels], rec[all_labels], f1_scores[all_labels] = spanwise_pr_re_f1(pred_seqs, gold_seqs)
    print("all-labels: Prec: %.3f, Rec: %.3f, F1: %.3f" % (prec[all_labels], rec[all_labels], f1_scores[all_labels]))

def pre_re_f1(gold_seqs,pred_seqs,label):
    pred_single_label = [[keep_label_of_interest(x,label) for x in sequence] for sequence in pred_seqs]
    gold_single_label = [[keep_label_of_interest(x,label) for x in sequence] for sequence in gold_seqs]
    return spanwise_pr_re_f1(pred_single_label, gold_single_label)

def compute_TP_P(guessed, correct):
    assert len(guessed) == len(correct)
    correctCount = 0
    count = 0

    idx = 0
    while idx < len(guessed):
        if guessed[idx][0] == 'B':  # A new chunk starts
            count += 1

            if guessed[idx] == correct[idx]:
                idx += 1
                correctlyFound = True

                while idx < len(guessed) and guessed[idx][0] == 'I':  # Scan until it no longer starts with I
                    if guessed[idx] != correct[idx]:
                        correctlyFound = False

                    idx += 1

                if idx < len(guessed):
                    if correct[idx][0] == 'I':  # The chunk in correct was longer
                        correctlyFound = False

                if correctlyFound:
                    correctCount += 1
            else:
                idx += 1
        else:
            idx += 1

    return correctCount,count

if __name__ == '__main__':
    gold = [
        ['O','O','O','B-ORG','O','O','O','B-PER','I-PER','O','O'],
        ['O','O','O','B-ORG','O','B-ORG','O','O','O','B-PER','I-PER','O','O']
    ]
    pred = [
        ['O','O','O','B-LOC','O','O','O','B-PER','I-PER','O','O'],
        ['O','O','O','B-ORG','O','B-ORG','O','O','O','B-PER','I-PER','O','O']
    ]
    p,r,f=spanwise_pr_re_f1(pred,gold)
    print(f)
    # calc_print_prefixed_seqtag_metrics(gold, pred, ['PER', 'ORG'])
    # tags = ['O','O','B-PER','I-PER','L-PER','O','U-ORG','O']
    # print(bilou2bio(tags))
    # print(bio_to_spans(bilou2bio(tags)))
    # tags = ['O','O','B-PER','U-PER','L-PER','O','U-ORG','O']
    # print(bilou2bio(tags))
    # print(bio_to_spans(bilou2bio(tags)))
    # tags = ['O','O','B-PER','O','L-PER','O','U-ORG','O']
    # print(bilou2bio(tags))
    # print(bio_to_spans(bilou2bio(tags)))
    # tags = ['O','O','B-PER','B-PER','L-PER','O','U-ORG','O']
    # print(bilou2bio(tags))
    # print(bio_to_spans(bilou2bio(tags)))