package tsml.transformers;

import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Interface for time series transformers.
 *
 * Previously, we have used weka filters for time series transforms. This redesign
 * is to make our code more like sktime and reduce our Weka dependency. The Filter
 * mechanism has some down sides
 *
 * Ultimately this will include the new data model
 *
 * @author Tony Bagnall 1/1/2020
 *
 */
public interface Transformer extends CapabilitiesHandler, ParamHandler {

    /**
     * perform the transform process. Some algorithms may require a fit before transform
     * (e.g. shapelets, PCA) others may not (FFT, PAA etc).
     * Should we throw an exception? Default to calling instance transform?
     * Need to determine where to setOut
     * @return Instances of transformed data
     */
    default Instances transform(Instances data){
        Instances output = determineOutputFormat(data);
        for(Instance inst : data){
            output.add(transform(inst));
        }
        return output;
    }

    Instance transform(Instance inst);

    Instances determineOutputFormat(Instances data) throws IllegalArgumentException ;

    default void setOptions(String[] options) throws Exception{
        //DEFAULT DOES NOTHING
    }

    //do a default capabilities that covers normal time series.
    default Capabilities getCapabilities(){
        Capabilities result = new Capabilities(this);
        result.disableAll();

        result.setMinimumNumberInstances(2);

        // attributes
        result.enable(Capabilities.Capability.RELATIONAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        return result;
    }
}
