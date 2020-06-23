package tsml.classifiers.distance_based.proximity;

import tsml.classifiers.distance_based.utils.collections.params.ParamSet;

public abstract class BaseTrainEstimateMethod implements TrainEstimateMethod {

    @Override public String toString() {
        return getClass().getSimpleName();
    }

    @Override public ParamSet getParams() {
        return new ParamSet();
    }
}
