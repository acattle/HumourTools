"""
Adapted from https://nbviewer.jupyter.org/github/lukewrites/NP_chunking_with_nltk/blob/master/NP_chunking_with_the_NLTK.ipynb
"""

import nltk

class NPChunker:
    def __init__(self):
        self.patterns = """
            NP: {<JJ>*<NN*>+}
            {<JJ>*<NN*><CC>*<NN*>+}
            """
        self.chunker = nltk.RegexpParser(self.patterns)
    
    def prepare_text(self, input):
        sentences = nltk.sent_tokenize(input)
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        sentences = [nltk.pos_tag(sent) for sent in sentences]
        sentences = [NPChunker.parse(sent) for sent in sentences]
        return sentences
    
    
    def parsed_text_to_NP(self, sentences):
        nps = []
        for sent in sentences:
            tree = NPChunker.parse(sent)
            for subtree in tree.subtrees():
                if subtree.node == 'NP':
                    t = subtree
                    t = ' '.join(word for word, tag in t.leaves())
                    nps.append(t)
        return nps
    
    
    def sent_parse(self, input):
        sentences = self.prepare_text(input)
        nps = self.parsed_text_to_NP(sentences)
        return nps
    
    def find_nps(self, text):
        prepared = self.prepare_text(text)
        parsed = self.parsed_text_to_NP(prepared)
        final = self.sent_parse(parsed)
        return final