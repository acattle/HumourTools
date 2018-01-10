'''
    Created on Jan 10, 2018
    
    :author: Andrew Cattle <acattle@cse.ust.hk>
    
    This module contains convenience functions for saving and loading
    sklearn.pipeline.Pipeline objects which contain KerasClassifier or
    KerasRegressor objects.
    
    Currently, the KerasClassifer and KerasRegressor wrappers do not support
    pickle/joblib/dill/etc but the wrapped Keras model can be saved directly
    using the model's save() function. See:
    
    https://github.com/keras-team/keras/issues/4274
    
    While this works for the model, it doesn't help with saving the fitted
    pipeline. However, we can save the model individually then remove it's
    reference from the pipeline, allowing us to pickle the pipeline as normal.
    Loading the model is as simple as loading the model and pipeline seperately
    and reassigning the reference. See:
    
    https://stackoverflow.com/questions/37984304/how-to-save-a-scikit-learn-pipline-with-keras-regressor-inside-to-disk/43415459
'''
from sklearn.externals import joblib
from keras.models import load_model

DEFAULT_MODEL_SUFFIX = "-model.h5"
DEFAULT_PIPELINE_SUFFIX = "-pipeline.pkl"

def save_keras_pipeline(file_loc, pipeline, estimator_name="estimator", model_file_suffix=DEFAULT_MODEL_SUFFIX, pipeline_file_suffix=DEFAULT_PIPELINE_SUFFIX):
    """
        Convenience method that allows for the saving of sklearn pipelines which
        contain KerasClassifier or KerasRegressor. Pipelines can then be
        reloaded using load_keras_pipeline().
        
        Models are saved as two separate files. By default:
            <file_loc>-model.h5
            <file_loc>-pipeline.pkl
        
        :param file_loc: the directory and filename prefix to use when saving the pipeline
        :type file_loc: str
        :param pipeline: the pipeline to be saved
        :type pipeline: sklearn.pipeline.Pipeline
        :param estimator_name: the name given to the KerasClassifier/KerasRegressor during pipeline construction
        :type estimator_name: str
        :param model_file_suffix: The suffix used when saving the Keras model (default = "-model.h5")
        :param model_file_suffix: str
        :param pipeline_file_suffix: The suffix used when saving the pipeline (default = "-pipeline.pkl")
        :param pipeline_file_suffix: str
    """
    #save the underlying Keras model
    model = pipeline.named_steps[estimator_name].model
    model.save("{}{}".format(file_loc, model_file_suffix))
    
    #remove reference to the actual Keras model so that we can pickle the pipeline
    pipeline.named_steps[estimator_name].model = None
    joblib.dump(pipeline, "{}{}".format(file_loc, pipeline_file_suffix))
    
    #replace the model so subsequent call to the pipeline don't fail
    #TODO: is this needed?
    pipeline.named_steps[estimator_name].model = model

def load_keras_pipeline(file_loc, estimator_name="estimator", model_file_suffix=DEFAULT_MODEL_SUFFIX, pipeline_file_suffix=DEFAULT_PIPELINE_SUFFIX):
    """
        Convenience method that allows for the loading of sklearn pipelines
        which contain KerasClassifier or KerasRegressor and have been saved
        using save_keras_pipeline().
        
        Note that save_keras_pipeline() saves models as two files and both files
        must be present. By default these files are:
            <file_loc>-model.h5
            <file_loc>-pipeline.pkl
        
        :param file_loc: the directory and filename prefix to use when loading the pipeline
        :type file_loc: str
        :param estimator_name: the name given to the KerasClassifier/KerasRegressor during pipeline construction
        :type estimator_name: str
        :param model_file_suffix: The suffix used when saving the Keras model (default = "-model.h5")
        :param model_file_suffix: str
        :param pipeline_file_suffix: The suffix used when saving the pipeline (default = "-pipeline.pkl")
        :param pipeline_file_suffix: str
        
        :returns: the loaded pipeline
        :rtype: sklearn.pipeline.Pipeline
    """
    #load the Keras model and the pipeline seperately
    model = load_model("{}{}".format(file_loc, model_file_suffix)) 
    pipeline = joblib.load("{}{}".format(file_loc, pipeline_file_suffix))
    
    #replace the model in the pipeline
    pipeline.named_steps[estimator_name].model = model
    
    return pipeline