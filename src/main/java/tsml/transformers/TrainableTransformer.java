package tsml.transformers;

import weka.core.Instances;
/**
 * Interface for time series transformers that require training, extending Transformer interface,
 * which does not require a fit stage
 *
 * @author Aaron Bostrom, Tony Bagnall
 * */
public interface TrainableTransformer extends Transformer {

    /**
     * Build the transform model from train data, storing the necessary info internally.
     * @param data
    */
    void fit(Instances data);

    /**
     * @return true if the the training (fitting) has already happened
     */
    boolean isFit();

    /**
     *
     * @param data
     * @return
     */
    default Instances fitTransform(Instances data){
        fit(data);
        return transform(data);
    }

    /**
     * main transform method. This automatically fits the model if it has not already been fit
     * it effectively implements fitAndTransform. The assumption is that if a Transformer is Trainable,
     * it is not usable until it has been trained/fit
     * @param data
     * @return
     */
    @Override
    default Instances transform(Instances data){
        if(!isFit())
            fit(data);

        return Transformer.super.transform(data);
    }
}