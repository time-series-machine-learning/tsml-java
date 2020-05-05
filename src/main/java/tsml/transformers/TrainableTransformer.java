package tsml.transformers;

import weka.core.Instances;

public interface TrainableTransformer extends Transformer {

    /**
     * Build the transform model from train data, storing the necessary info internally.
     * @param data
    */
    void fit(Instances data);
        /**
     * fits and transforms, default to calling fit then transform!
     * @return
     */

    boolean isFit();

    default Instances fitTransform(Instances data){
        fit(data);
        return transform(data);
    }
}