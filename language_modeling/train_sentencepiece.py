import sentencepiece as spm

if __name__ == '__main__':
    spm.SentencePieceTrainer.Train(
        '--input=/tmp/text.txt --model_prefix=tilos_test --vocab_size=3100 --character_coverage=0.9999 --model_type=bpe --input_sentence_size=100000000 --shuffle_input_sentence=true')

    sp = spm.SentencePieceProcessor()
    sp.Load("language_modeling/tilos_test.model")
    print(sp.EncodeAsPieces("This is a test"))
    print(sp.NBestEncodeAsPieces("This is a test", 5))
    print(sp.EncodeAsIds("This is a test"))