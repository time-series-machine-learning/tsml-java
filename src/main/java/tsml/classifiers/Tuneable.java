/*
 Tunable interface enforces the method getDefaultParameterSearchSpace, for use 
with the general TunedClassifier class. 

ParameterSpace are created by calls to the ParameterSpace object with 
addParameter(String name, values), where values can be arrays or a List.


 */
package tsml.classifiers;

import evaluation.tuning.ParameterSpace;

/**
 * For classifiers which can be tuned, requires an overidden setOptions from abstract classifier in most cases.
 *
 * @author ajb
 */
public interface Tuneable {
    
    /**
     * getDefaultParameterSearchSpace returns the possible parameter values
     * that can be looked for with the TunedClassifier
     *
     * @return default parameter space for tuning
     */
    ParameterSpace getDefaultParameterSearchSpace();
}
