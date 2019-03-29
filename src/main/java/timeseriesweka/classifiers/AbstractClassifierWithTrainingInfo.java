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

import weka.classifiers.AbstractClassifier;
import evaluation.storage.ClassifierResults;
import weka.core.Capabilities;

/**
 *
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

AbstractClassifierWithTrainingInfo c= //Get classifier
c.buildClassifier(train)    //ALL STATS SET HERE
* 
 * @author ajb
 */
abstract public class AbstractClassifierWithTrainingInfo extends AbstractClassifier implements SaveParameterInfo{
    protected boolean fullyNestedEstimates=true;
    protected ClassifierResults trainResults =new ClassifierResults();
   
     @Override
    public String getParameters() {
        return "FullyNestedEstimates,"+fullyNestedEstimates;
    }
     
    public String getTrainInfo() {
        return getParameters();
    }
/**
   * Returns default capabilities of the classifier. These are that the 
   * data must be numeric, with no missing and a nominal class
   * @return the capabilities of this classifier
**/    
    @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();
    // attributes must be numeric
    // Here add in relational when ready
    result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
    // class
    result.enable(Capabilities.Capability.NOMINAL_CLASS);
    // instances
    result.setMinimumNumberInstances(0);   
    return result;
  }
     
}
