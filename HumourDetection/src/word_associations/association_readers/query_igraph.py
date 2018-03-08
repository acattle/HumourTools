'''
Created on Feb 8, 2018

@author: Andrew

This is a script for running igraph via WSL
'''
from tempfile import TemporaryDirectory
import os
import subprocess

def get_strengths_wsl(word_pairs, dataset, pajek_loc, tmpdir=None):
    cmd = f"wsl python3 /mnt/d/git/HumourDetection/HumourDetection/src/word_associations/association_readers/igraph_readers.py {dataset} {pajek_loc}"
    with TemporaryDirectory(dir=tmpdir) as td:
        wp_loc = os.path.join(td, "word_pairs")
        out_loc = os.path.join(td, "strength_output")
        
        with open(wp_loc, "w") as wp_f:
            for a,b in word_pairs:
                wp_f.write(f"{a}\t{b}\n")
        
        with open(wp_loc, "r") as wp_f, open(out_loc, "w") as out_f:
            #TODO: will this properly report errors?
            subprocess.run(cmd, stdin=wp_f, stdout=out_f)
        
        strengths=[]
        with open(out_loc, "r") as out_f:
            for line in out_f:
#                 if line.strip():
                strengths.append(float(line))
    
    if (len(word_pairs) != len(strengths)):
        raise RuntimeError("For some reason we didn't get a strenght for each unique word pair. Check error above")
    
    return strengths