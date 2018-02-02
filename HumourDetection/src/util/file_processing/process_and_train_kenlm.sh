#!/bin/bash

#mkdir /mnt/c/Users/Andrew/Downloads/news-discuss-models
cd /mnt/c/Users/Andrew/Downloads/news-discuss-models
#mkdir tmp
#python3 tokenize_file.py -i news-discuss.en.txt.gz.part -o news-discuss_tokenized.txt
/mnt/c/Users/Andrew/kenlm/build/bin/lmplz -o 4 -S 80% -T ./tmp --prune 0 3 <news-discuss_tokenized.txt | /mnt/c/Users/Andrew/kenlm/build/bin/build_binary trie /dev/stdin news-discuss_4.bin
/mnt/c/Users/Andrew/kenlm/build/bin/lmplz -o 3 -S 80% -T ./tmp --prune 0 3 <news-discuss_tokenized.txt | /mnt/c/Users/Andrew/kenlm/build/bin/build_binary trie /dev/stdin news-discuss_3.bin
/mnt/c/Users/Andrew/kenlm/build/bin/lmplz -o 2 -S 80% -T ./tmp --prune 0 3 <news-discuss_tokenized.txt | /mnt/c/Users/Andrew/kenlm/build/bin/build_binary trie /dev/stdin news-discuss_2.bin
/mnt/c/Users/Andrew/kenlm/build/bin/lmplz -o 1 -S 80% -T ./tmp --prune <news-discuss_tokenized.txt >news-discuss_1.arpa
