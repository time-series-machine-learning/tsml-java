package tsml.transformers;

import org.apache.commons.lang3.NotImplementedException;
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
 * @author Tony Bagnall 1/1/2020, Aaron Bostrom
 *
 */
public interface Transformer extends CapabilitiesHandler{

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

    /**
     * Transform a new instance into the format described in determineOutputFormat
     * @param Instance inst
     * @return transformed Instance
     */
    Instance transform(Instance inst);

    /**
     * Method that constructs a holding Instances for transformed data, without doing the transform
     * @param Instances data to derive the format from
     * @return empty Instances to hold transformed Instance objects
     * @throws IllegalArgumentException
     */
    Instances determineOutputFormat(Instances data) throws IllegalArgumentException;

    /**
     * setOptions can be implemented and used in the Tuning mechanism that classifiers employ.
     * It does not have to be implemented though, so there is a default method that throws an exception
     * if it is called
     * @param String array of options that should be in the format of flag,value
     * @throws Exception
     */
    default void setOptions(String[] options) throws Exception{
        throw new NotImplementedException("calling default method of setOptions in Transformer interface, it has not been implemented for class "+this.getClass().getSimpleName());
    }

    /**
     * getCapabilities determines what type of data this Transformer can handle. Default is
     * all numeric  attributes, nominal class, no missing values
     * @return
     */
    default Capabilities getCapabilities(){
        Capabilities result = new Capabilities(this);
        result.disableAll();

        result.setMinimumNumberInstances(2);
        // attributes
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        return result;
    }
}
