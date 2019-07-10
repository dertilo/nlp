import shlex
import sys

from flair.embeddings import FlairEmbeddings
from flair.models import LanguageModel

# char_lm_embeddings = FlairEmbeddings('news-forward')
def interactive(lang_model_file):
    lm = LanguageModel.load_language_model(lang_model_file)
    import argparse

    parser = argparse.ArgumentParser(description='Argparse Test script')
    parser.add_argument("input", help='some parameter',default='This is a test')
    parser.add_argument("--temperature", help='some parameter',default=None,required=False)
    parser.add_argument("--seqlen", help='some parameter',default=None,required=False)
    seqlen, temperature = 20,1.0
    while (1):
        inp = input('type input: > ')
        if inp == 'q' or inp == 'quit': break
        args,_ = parser.parse_known_args(shlex.split(inp))
        seqlen = seqlen if args.seqlen is None else int(args.seqlen)
        temperature = temperature if args.temperature is None else float(args.temperature)

        inp = args.input
        text, likelihood = lm.generate_text(inp, number_of_characters=seqlen, temperature=temperature)
        print(text)



if __name__ == '__main__':
    interactive(sys.argv[1])