from numpy import require
import sentencepiece as sp 
import argparse

def main():
    print('=> Subword training....')
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i','--input_file', type=str, required=True, dest='input', help='Input file name')
    parser.add_argument('-o','--output_file', type=str, required=True, dest='output', help='Output file name')
    parser.add_argument('-v','--vocab_size', type=str,required=True, dest='vocab', help='Vocabulary size')

    args = parser.parse_args()
    sp.SentencePieceTrainer.Train('--input=' + args.input + ' --model_prefix=' + args.output + ' --vocab_size=' + args.vocab
                                    + ' --hard_vocab_limit=false')

    print('=> Done.')

if __name__ == '__main__':
    main()