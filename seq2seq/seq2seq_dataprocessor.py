import random
import re
import torch

from commons import data_io, util_methods

from seq2seq.getting_processing_data import Voc, UNK_token, EOS_token, zeroPadding, binaryMatrix
from text_classification.classifiers.common import DataProcessorInterface
GAP = '_'
gap_symbol_len=1 # just one integer number

class Seq2GapDataProcessor(DataProcessorInterface):


    def __init__(self,
                 corpus_name="sci-keyword-sentencs",
                seq_len=20
                 ) -> None:
        super().__init__()
        self.seq_len=seq_len
        self.voc=Voc(corpus_name)

    def fit(self, data_path):
        line_g = data_io.read_lines_from_files(data_path, limit=100)
        [self.voc.addWord(char) for line in line_g for char in line]
        self.voc.trim(10)
        self.voc.addWord(GAP)
        assert '#' in self.voc.word2index.keys()
        print("Counted words:", self.voc.num_words)
        return self

    def transform(self, data):
        inputts=[]
        for sample in data:
            assert len(sample.split(GAP))==2
            left_context, right_context = sample.split(GAP)
            inputts.append([self.voc.word2index[symbol] for symbol in list(left_context) + [GAP] + list(right_context)])

        return self.tensorfy_input(inputts)


    def tensorfy(self,in_seq_batch, out_seq_batch):
        input_variable, lengths = self.tensorfy_input(in_seq_batch)

        max_target_len = max([len(indexes) for indexes in out_seq_batch])
        padList = zeroPadding(out_seq_batch)
        mask = binaryMatrix(padList)
        mask = torch.ByteTensor(mask)
        target_variable = torch.LongTensor(padList)
        return input_variable, lengths, target_variable, mask, max_target_len

    def tensorfy_input(self, in_seq_batch):
        lengths = torch.tensor([len(indexes) for indexes in in_seq_batch])
        padList = zeroPadding(in_seq_batch)
        input_variable = torch.LongTensor(padList)
        return input_variable, lengths

    def build_get_batch_fun(self, data_source, batch_size):
        def normalizeString(s):
            s = re.sub(r"\s+", r" ", s).strip()
            s = ''.join([c if c in self.voc.word2index else self.voc.index2word[UNK_token] for c in s])
            return s

        def build_batch_generator(batch_size):
            def replace_if_gap(char:str):
                return '-' if char==GAP else char
            def carve_a_gap(char_g):
                gap_len = random.randint(2, 9)
                start_ind = random.randint(0, self.seq_len-gap_len-gap_symbol_len)
                left_context = [replace_if_gap(next(char_g)) for _ in range(start_ind)]
                gap = [replace_if_gap(next(char_g)) for _ in range(gap_len)]
                right_context = [replace_if_gap(next(char_g)) for _ in range(self.seq_len-start_ind-gap_symbol_len)]
                inputt = [self.voc.word2index[symbol] for symbol in left_context+[GAP]+right_context]
                outputt = [self.voc.word2index[symbol] for symbol in gap]+[EOS_token]
                assert len(inputt)==self.seq_len
                return inputt,outputt

            def carve_gaps(paragraph):
                char_g = (char for char in normalizeString(paragraph))
                while True:
                    try:
                        yield carve_a_gap(char_g)
                    except StopIteration:
                        break

            paragraph_g = data_io.read_lines_from_files(data_source)
            input_output_g = (in_out for p in paragraph_g for in_out in carve_gaps(p))
            return util_methods.iterable_to_batches(input_output_g, batch_size)

        batch_g = [build_batch_generator(batch_size)]

        def get_batch(message=None):
            try:
                batch = next(batch_g[0])
                if len(batch)!=batch_size:
                    raise StopIteration
            except StopIteration:
                batch_g[0] = build_batch_generator(batch_size)
                raise StopIteration

            in_seq_batch = [s for s,_ in batch]
            out_seq_batch = [s for _,s in batch]

            return self.tensorfy(in_seq_batch,out_seq_batch)

        return get_batch

if __name__ == '__main__':
    data_path='/home/tilo/gunther/arxiv_papers/ml_nlp_parsed'
    dp = Seq2GapDataProcessor()
    dp.fit(data_path)
    get_batch_fun=dp.build_get_batch_fun(data_path,batch_size=32)
    batch = get_batch_fun(None)
    print()
