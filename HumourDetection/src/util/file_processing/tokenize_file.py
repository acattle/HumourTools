'''
Created on Jan 31, 2018

@author: Andrew Cattle <acattle@connect.ust.hk>

A quick script for reading in a document and tokenizing it.
'''
import argparse
from nltk import word_tokenize

def tokenize_file(input_loc, output_loc, tokenizer=None, lower=True):
    """
        Tokenize file at input_loc and write it to output_loc
        
        :param input_loc: the location of the corpus
        :type input_loc: str
        :param output_loc: the output locations
        :type output_loc: str
        :param tokenizer: tokenization function to use. If None, defaults to nltk.word_tokenize()
        :type tokenizer: Callable[[str], List[str]]
        :param lower: whether the corpus whould be lowercased or not
        :type lower: bool
    """
    
    tokenize = tokenizer if tokenizer else word_tokenize #If tokenizer is None, default to word_tokenize
    
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
    
    tokenize_file(args.input, args.output)
    
    