'''
Created on May 23, 2019

:author: Andrew Cattle <acattle@connect.ust.hk>

This module contains code for reading the Castro et al. (2018) dataset used in
the HAHA 2019 group task. The dataset consists of Spanish tweets which have
been human annotated for perceived humour.

For more information see:

    Castro, S., Chiruzzo, L., Rosa, A., Garat, D., & Moncecchi, G. (2018). A
    crowd-annotated spanish corpus for humor analysis. In Proceedings of the
    Sixth International Workshop on Natural Language Processing for Social
    Media (pp. 7-11).
    
    https://competitions.codalab.org/competitions/22194
'''

import csv

def read_haha2019_file(file_loc, header=True, test=False, encoding="utf-8"):
    """
    Read a file in Castro et al. (2018) format. File should be in CSV format
    where each line corresponds to one tweet and its humour labels.
    
    Expected line format is:
    <id>, <tweet body>, <binary humour label>, <total # of ratings>, <# of 1 ratings>, <# of 2 ratings>, <# of 3 ratings>, <# of 4 ratings>, <# of 5 ratings>, <average rating>
    
    :param file_loc: location of file in Castro et al. (2018) format
    :type file_loc: str
    :param header: specifies whether first line is header
    :type header: bool
    :param test: specifies whether file is in test format (i.e. only <id> and <tweet body>, no humour ratings)
    :type test: bool
    :param encoding: encoding of the file specified by file_loc. Default is "utf-8"
    :type encoding: str
    
    :returns: the tweet id, tweet body with the binary label and fine grained label
    :rtype: List[Tuple[str, str, int, float]]
    """
    
    with open(file_loc, "r", encoding=encoding) as f:
        r = csv.reader(f)
        
        if header:
            next(r) #pop the header row
        
        documents=[]
        for row in r:
            
            if test:
                id_num, tweet = row
                #no ratings included in test format. Default to Nones
                bin_label, num_ratings, num_1s, num_2s, num_3s, num_4s, num_5s, avg_rating = None, None, None, None, None, None, None, None
                
            else:
                id_num, tweet, bin_label, num_ratings, num_1s, num_2s, num_3s, num_4s, num_5s, avg_rating = row
                
                bin_label = int(bin_label)
                
                if avg_rating:
                    if avg_rating == "NULL":
                        avg_rating = 0.0
                    else:
                        avg_rating = float(avg_rating)
            
            documents.append((id_num, tweet, bin_label, avg_rating))
        
        return documents
            


if __name__ == "__main__":
    training_loc = "d:/datasets/HAHA 2019/haha_2019_train.csv"
    test_loc = "d:/datasets/HAHA 2019/haha_2019_test.csv"
    output_loc = "d:/datasets/HAHA 2019/test_output.csv"
    
    training = read_haha2019_file(training_loc)
    test = read_haha2019_file(test_loc, test=True)
    
    training_ids, training_docs, training_labels, _ = zip(*training)
    test_ids, test_docs, _, _ = zip(*test)
    
    all_docs = training_docs + test_docs
    tensors, _ = decompose_tensors(all_docs, win_size=5, cp_rank=50)
    training_tensors = tensors[:len(training_docs)]
    test_tensors = tensors[len(training_docs):]
    
    
    label_prop_model = LabelSpreading()
    label_prop_model.fit(training_tensors, training_labels)
    label_predictions = label_prop_model.predict(test_tensors)
    
    
    with open(output_loc, "w") as o_f:
        writer = csv.writer()
        
        #write headerz
        writer.writerow(["id","is_humor","funniness_average"])
        
        for test_id, label_prediction in zip(test_ids, label_predictions):
            writer.writerow([test_id, label_prediction])