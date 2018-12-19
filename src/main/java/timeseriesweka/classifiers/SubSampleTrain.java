/*
Indicates that the class can subsample the train set if the option is set
 */
package timeseriesweka.classifiers;

import utilities.InstanceTools;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public interface SubSampleTrain {

    public void subSampleTrain(double prop, int seed);
    default Instances subSample(Instances full, double proportion, int seed){
        return InstanceTools.subSampleFixedProportion(full, proportion, seed);
    }
}
