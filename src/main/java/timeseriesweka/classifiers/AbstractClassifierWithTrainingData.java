/*
@author: ajb

Extends the AbstractClassifier to store information about the training phase of 
the classifier. The minimium any classifier that extends this should store
is the build time in buildClassifier, through calls to System.currentTimeMillis()  
at the start and end. 

the method getParameters() can be enhanced to include any parameter info for the 
final classifier. getParameters() is called to store information on the second line
of file storage format testFoldX.csv.

ClassifierResults trainResults can also store other information about the training,
including estimate of accuracy, predictions and probabilities. NOTE that these are 
assumed to be set through nested cross validation in buildClassifier or through
out of bag estimates where appropriate. IT IS NOT THE INTERNAL TRAIN ESTIMATES.

If the classifier performs some internal parameter optimisation, then ideally 
there should be another level of nesting to get the estimates. IF THIS IS NOT DONE,
SET THE VARIABLE fullyNestedEstimates to false. The user can do what he wants 
with that info

Also note: all values in trainResults are set without any reference to the train 
set at all. All the variables for trainResults are set in buildClassifier, which 
has no access to test data at all. It is completely decoupled. 

Instances train=//Get train

AbstractClassifierWithTrainingData c= //Get classifier
c.buildClassifier(train)    //ALL STATS SET HERE


 */
package timeseriesweka.classifiers;

import utilities.SaveParameterInfo;
import weka.classifiers.AbstractClassifier;
import utilities.ClassifierResults;

/**
 *
 * @author ajb
 */
abstract public class AbstractClassifierWithTrainingData extends AbstractClassifier implements SaveParameterInfo{
    protected boolean fullyNestedEstimates=true;
    protected ClassifierResults trainResults =new ClassifierResults();
   
     @Override
    public String getParameters() {
        return "BuildTime,"+trainResults.buildTime;
    }
     
    public String getTrainData() {
        StringBuilder sb=new StringBuilder("AccEstimateFromTrain,");
        sb.append(trainResults.acc).append(",");
        
        return "BuildTime,"+trainResults.buildTime;
    }
     
}
