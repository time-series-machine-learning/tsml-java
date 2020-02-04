package tsml.classifiers;

import utilities.StopWatch;

public interface TrainTimeable {
    default long getTrainTimeNanos() { return -1; };
    default StopWatch getTrainTimer() {
        throw new UnsupportedOperationException();
    }
    default StopWatch getTrainEstimateTimer() {
        throw new UnsupportedOperationException();
    }
}
