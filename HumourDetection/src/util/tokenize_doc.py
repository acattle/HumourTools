'''
Created on Jan 31, 2018

@author: Andrew
'''
import argparse
from nltk import word_tokenize

def tokenize_document(input_loc, output_loc, tokenize=None, lower=True):
    if not tokenize:
        tokenize = word_tokenize
    
    with open(input_loc, "r") as input_f, open(output_loc, "w") as output_f:
        for l in input_f:
            
            if lower:
                l=l.lower()
            
            tokens = tokenize(l)
            
            tokenized_string = " ".join(tokens)
            
            output_f.write(tokenized_string)
            output_f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Document tokenizer')
    parser.add_argument('--input', '-i', help='input file', required=True)
    parser.add_argument('--output', '-o', help='output file', required=True)
    args = parser.parse_args()
    
    tokenize_document(args.input, args.output)
    
    