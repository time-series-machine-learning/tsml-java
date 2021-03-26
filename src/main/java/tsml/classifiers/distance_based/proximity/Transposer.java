package tsml.classifiers.distance_based.proximity;

import tsml.data_containers.TimeSeriesInstance;
import tsml.transformers.Transformer;
import weka.core.Instances;

public class Transposer implements Transformer {
    @Override public Instances determineOutputFormat(final Instances data) throws IllegalArgumentException {
        return data; // todo transpose it
    }

    @Override public TimeSeriesInstance transform(final TimeSeriesInstance inst) {
        return new TimeSeriesInstance(inst.toTransposedArray(), inst.getLabelIndex());
    }
}
