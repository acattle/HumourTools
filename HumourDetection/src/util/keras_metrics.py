'''
Created on Feb 13, 2018

:author: Andrew Cattle <acattle@cse.ust.hk>
'''

from keras import backend as K

def r2_score(y_true, y_pred):
    """
    Calaculates R2 score according to
    http://jmbeaujour.com/ml/2017/03/20/CoeffDetermination_CustomMetric4Keras/
    """
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def pearsonr(y_true, y_pred):
    """
    Calaculates Pearson's R
    """
    #https://stackoverflow.com/questions/46115896/keras-handling-batch-size-dimension-for-custom-pearson-correlation-metric
    fsp = y_pred - K.mean(y_pred) #being K.mean a scalar here, it will be automatically subtracted from all elements in y_pred
    fst = y_true - K.mean(y_true)
    
    devP = K.std(y_pred)
    devT = K.std(y_true)
    
    return K.mean(fsp*fst)/(devP*devT)