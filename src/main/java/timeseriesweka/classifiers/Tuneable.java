/*
 Tunable interface enforces the method getDefaultParameterSearchSpace, for use 
with the general TunedClassifier class. 

ParameterSpace are created by calls to the ParameterSpace object with 
addParameter(String name, values), where values can be arrays or a List.


 */
package timeseriesweka.classifiers;

import evaluation.tuning.ParameterSpace;

/**
 *
 * @author ajb
 */
public interface Tuneable {
    
    /**
     * getDefaultParameterSearchSpace returns the possible parameter values
     * that can be looked for with the TunedClassifier
     * @return 
     */
    ParameterSpace getDefaultParameterSearchSpace();
}
