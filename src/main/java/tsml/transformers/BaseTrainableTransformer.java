package tsml.transformers;

import weka.core.Instance;
import weka.core.Instances;

public abstract class BaseTrainableTransformer implements TrainableTransformer {

    private boolean isFit;

    @Override public void fit(final Instances data) {
        isFit = true;
    }

    public void reset() {
        isFit = false;
    }

    @Override public boolean isFit() {
        return isFit;
    }

}
