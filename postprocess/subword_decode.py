import sentencepiece 
import argparse

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i','--input_file', type=str, required=True, dest='input', help='Input file name')
    parser.add_argument('-o','--output_file', type=str, required=True, dest='output', help='Output file name')
    parser.add_argument('-m','--model', type=str,required=True, dest='model', help='Model name')

    args = parser.parse_args()
    sp = sentencepiece.SentencePieceProcessor(model_file=args.model)
    doc_parsed = []

    print('=> Subword decoding....')
    with open(args.input, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip()
            pieces = ' '.join(sp.decode(line.split(' ')))
            doc_parsed.append(pieces)
        fp.close()

    with open(args.output, 'w', encoding='utf-8') as fp:
        for line in doc_parsed:
            fp.write(line + '\n')
        fp.close()
    
    print('=> Done.')

if __name__ == '__main__':
    main()