from collections import Counter
from pprint import pprint

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from getting_data.clef2019 import get_Clef2019_data


def aggregate_utterances_to_turns(utterances):
    turns = []
    current_speaker = ''
    turn = None
    for utt in utterances:
        if current_speaker != utt['speaker']:
            if turn is not None:
                turns.append({'debatefile':turn[0]['debatefile'],
                              'utterances':[u['utterance'] for u in turn],
                              'speaker':turn[0]['speaker']})
            turn = [utt]
            current_speaker = utt['speaker']
        else:
            turn.append(utt)
    return turns

if __name__ == '__main__':
    path = 'somewhere/clef2019-factchecking-task1/data/training'
    utterances = get_Clef2019_data(path)
    turns = aggregate_utterances_to_turns(utterances)
    counter = Counter([t['speaker'] for t in turns])
    pprint(counter)
    turns = [t for t in turns if counter[t['speaker']]>5]


    splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=111)
    splits = [(train, test) for train, test in
              splitter.split(X=range(len(turns)),y=[t['speaker'] for t in turns])]

    pipeline = Pipeline([
        ('ngrams', TfidfVectorizer(ngram_range=(1, 1))),
        ('clf', SGDClassifier(alpha=.00001, loss='log', penalty="elasticnet", l1_ratio=0.2))
    ])

    train_idx, test_idx = splits[0]
    train_turns = [turns[i] for i in train_idx]
    test_turns = [turns[i] for i in test_idx]

    train_texts = [' '.join(turn['utterances']) for turn in train_turns]
    test_texts = [' '.join(turn['utterances']) for turn in test_turns]

    label_encoder = LabelEncoder()
    train_speakers = [turn['speaker'] for turn in train_turns]
    test_speakers = [turn['speaker'] for turn in test_turns]
    label_encoder.fit(train_speakers+test_speakers)

    encoded_train_targets = label_encoder.transform(train_speakers)
    encoded_test_targets = label_encoder.transform(test_speakers)

    pipeline.fit(train_texts, encoded_train_targets)

    pred_train = pipeline.predict(train_texts)
    pred_test = pipeline.predict(test_texts)
    print('TRAIN')
    print(metrics.classification_report(encoded_train_targets,pred_train,output_dict=False,target_names=label_encoder.classes_.tolist()))
    print('TEST')
    print(metrics.classification_report(encoded_test_targets,pred_test,output_dict=False,target_names=label_encoder.classes_.tolist()))

    '''
    expected output: 
    
        Counter({'TRUMP': 1171,
             'SYSTEM': 825,
             'CLINTON': 465,
             'QUESTION': 289,
             'SANDERS': 251,
             'BLITZER': 226,
             'PENCE': 210,
             'KAINE': 190,
             'WALLACE': 126,
             'QUIJANO': 108,
             'RUBIO': 101,
             'HOLT': 98,
             'CRUZ': 82,
             'COOPER': 75,
             'TODD': 69,
             'BASH': 63,
             'RADDATZ': 63,
             'PELOSI': 58,
             'MADDOW': 42,
             'SCHUMER': 38,
             'KASICH': 30,
             'CELESTE': 26,
             'HEWITT': 26,
             'LOUIS': 23,
             'CARSON': 17,
             'AUDIENCE': 8,
             'MR. SCHWAB': 7,
             'UNIDENTIFIED MALE': 5,
             'So as I look at what the president it doing, it adds up to me. We just have to keep -- try to get more support for those people on the ground in Syria and Iraq who have to actually physically take the territory back. MADDOW': 1,
             'SEN. BERNIE SANDERS (D-VT), PRESIDENTIAL CANDIDATE': 1,
             'HILLARY CLINTON (D-NY), FORMER SECRETARY OF STATE, PRESIDENTIAL CANDIDATE': 1,
             'AUDIENCE MEMBER': 1,
             '(APPLAUSE) CLINTON': 1,
             '(APPLAUSE) SANDERS': 1,
             'SYSYTEM': 1,
             'AUDIENCE MEMBERS': 1})

    TRAIN

                  precision    recall  f1-score   support
    
        AUDIENCE       1.00      1.00      1.00         6
            BASH       0.65      0.78      0.71        50
         BLITZER       0.73      0.91      0.81       181
          CARSON       1.00      1.00      1.00        14
         CELESTE       1.00      0.86      0.92        21
         CLINTON       0.95      0.92      0.93       372
          COOPER       0.94      0.77      0.84        60
            CRUZ       0.95      0.95      0.95        66
          HEWITT       1.00      0.62      0.76        21
            HOLT       0.98      0.83      0.90        78
           KAINE       0.95      0.89      0.92       152
          KASICH       1.00      0.92      0.96        24
           LOUIS       1.00      0.56      0.71        18
          MADDOW       1.00      0.79      0.89        34
      MR. SCHWAB       1.00      1.00      1.00         6
          PELOSI       0.97      0.78      0.87        46
           PENCE       0.97      0.90      0.93       168
        QUESTION       0.98      0.90      0.94       231
         QUIJANO       0.89      0.95      0.92        86
         RADDATZ       0.93      0.74      0.82        50
           RUBIO       0.99      0.98      0.98        81
         SANDERS       0.86      0.92      0.89       201
         SCHUMER       0.48      0.83      0.61        30
          SYSTEM       1.00      1.00      1.00       660
            TODD       0.98      0.84      0.90        55
           TRUMP       0.93      0.97      0.95       937
         WALLACE       0.82      0.74      0.78       101
    
       micro avg       0.92      0.92      0.92      3749
       macro avg       0.92      0.86      0.89      3749
    weighted avg       0.93      0.92      0.92      3749
    
    TEST
                  precision    recall  f1-score   support
    
        AUDIENCE       1.00      1.00      1.00         2
            BASH       0.29      0.31      0.30        13
         BLITZER       0.60      0.64      0.62        45
          CARSON       0.00      0.00      0.00         3
         CELESTE       0.00      0.00      0.00         5
         CLINTON       0.48      0.60      0.54        93
          COOPER       0.62      0.33      0.43        15
            CRUZ       0.53      0.50      0.52        16
          HEWITT       1.00      0.60      0.75         5
            HOLT       0.64      0.35      0.45        20
           KAINE       0.54      0.39      0.45        38
          KASICH       1.00      0.33      0.50         6
           LOUIS       1.00      0.20      0.33         5
          MADDOW       0.50      0.12      0.20         8
      MR. SCHWAB       0.00      0.00      0.00         1
          PELOSI       0.67      0.33      0.44        12
           PENCE       0.42      0.33      0.37        42
        QUESTION       0.56      0.60      0.58        58
         QUIJANO       0.65      0.68      0.67        22
         RADDATZ       0.67      0.62      0.64        13
           RUBIO       0.58      0.55      0.56        20
         SANDERS       0.40      0.34      0.37        50
         SCHUMER       0.00      0.00      0.00         8
          SYSTEM       1.00      0.99      1.00       165
            TODD       0.40      0.14      0.21        14
           TRUMP       0.63      0.82      0.71       234
         WALLACE       0.17      0.08      0.11        25
    
       micro avg       0.64      0.64      0.64       938
       macro avg       0.53      0.40      0.44       938
    weighted avg       0.62      0.64      0.62       938

    '''

