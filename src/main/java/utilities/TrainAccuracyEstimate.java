package utilities;

import weka.core.Instances;

/**
* Classifiers implementing this interface will perform a CV on the train data
* and implement a means of storing train predictions and probabilities. 
* 
* there are two use cases
 
 1. Just get it to write the train results to file
 c.writeCVTrainToFile("c:\temp\TrainFold1.csv");
 whether it writes predictions is classifier specific, see below. 
 
 2. Recover the train results in a ClassifierResults object. James to sort this out
* 
 * @author ajb
 */
public interface TrainAccuracyEstimate {

    
    void setFindTrainAccuracyEstimate(boolean setCV);
    
    /**
 *  classifiers implementing this interface can perform a CV
 * on the train data and store that data in a ClassifierResults object
     * @return true if this classifier actually finds the estimate
 */
    default boolean findsTrainAccuracyEstimate(){ return false;}
/**
 * TrainCV results are not by default written to file. If this method is called
 * they will be written in standard format, as defined in the ClassifierResults class
 * The minimum requirements for the train results are
 * 
 * ProblemName,ClassifierName,train
*  Parameter info, if available
*  TrainAccuracy
* If available, the preds and probs will also be written 
* Case1TrueClass,Case1PredictedClass,,ProbClass1,ProbClass2, ...
* Case2TrueClass,Case2PredictedClass,,ProbClass1,ProbClass2, ...
* 
 * @param train: Full file name for the TrainCV results
 */    
    void writeCVTrainToFile(String train);
/**
 * 
     * @return All the data from the train CV
    */
    ClassifierResults getTrainResults();    
    default int setNumberOfFolds(Instances data){
        return 10;
    }



}
