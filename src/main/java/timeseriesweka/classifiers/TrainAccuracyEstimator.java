/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package timeseriesweka.classifiers;

import evaluation.storage.ClassifierResults;
import weka.core.Instances;

/**
* Classifiers implementing this interface will perform a CV on the train data
* and implement a means of storing train predictions and probabilities. 
* 
* there are two use cases
 
* 
 * @author ajb
 */
public interface TrainAccuracyEstimator{

    
    void setFindTrainAccuracyEstimate(boolean setCV);
    
    /**
 *  classifiers implementing this interface can estimate probabilities and classes
 * on the train data and store that data in a ClassifierResults object. 
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
*  TrainAccuracy, build time, test time.
* If available, the preds and probs will also be written 
* Case1TrueClass,Case1PredictedClass,,ProbClass1,ProbClass2, ...
* Case2TrueClass,Case2PredictedClass,,ProbClass1,ProbClass2, ...
* 
 * @param train: Full file name for the TrainCV results
 */    
    void writeTrainEstimatesToFile(String train);

    default int setNumberOfFolds(Instances data){
        return data.numInstances()<10?data.numInstances():10;
    }

}
