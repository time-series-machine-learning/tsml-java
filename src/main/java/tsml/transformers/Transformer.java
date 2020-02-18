package tsml.transformers;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Interface for time series transformers.
 *
 * Previously, we have used weka filters for time series transforms. This redesign
 * is to make our code more like sktime and reduce our Weka dependency. The Filter
 * mechanism has some down sides
 *
 * Ultimately this will move away from instances too
 */
public interface Transformer {
    /**
     * Build the transform model from train data, storing the necessary info internally
     * @param data
     */
    void fit(Instances data);
    /**
     * perform the transform process. Some algorithms may require a fit before transform
     * (e.g. shapelets, PCA) others may not (FFT, PAA etc).
     * Should we throw an exception? Default to calling instance transform?
     * Need to determine where to setOut
     * @return Instances of transformed data
     */
    Instances transform(Instances data);
    Instance transform(Instance inst);
    /**
     * fits and transforms, default to calling fit then transform!
     * @return
     */
    default Instances fitTransform(Instances data){
        fit(data);
        return transform(data);
    }
    Instances determineOutputFormat(Instances data) throws IllegalArgumentException ;
}
