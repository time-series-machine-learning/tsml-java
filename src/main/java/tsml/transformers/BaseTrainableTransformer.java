package tsml.transformers;

import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import weka.core.Instance;
import weka.core.Instances;

public abstract class BaseTrainableTransformer implements TrainableTransformer {

    protected boolean fitted;

    @Override public void fit(final TimeSeriesInstances data) {
        fitted = true;
    }

    public void reset() {
        fitted = false;
    }

    @Override public boolean isFit() {
        return fitted;
    }

    @Override public Instance transform(final Instance inst) {
        return Converter.toArff(transform(Converter.fromArff(inst)));
    }

    @Override public void fit(final Instances data) {
        fit(Converter.fromArff(data));
    }
}
