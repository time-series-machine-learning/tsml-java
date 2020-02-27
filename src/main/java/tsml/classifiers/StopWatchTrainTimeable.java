package tsml.classifiers;

import utilities.stopwatch.StopWatch;

/**
 * Simple interface to defer TrainTimeable to StopWatch implementations.
 */
public interface StopWatchTrainTimeable extends TrainTimeable {
    default StopWatch getTrainTimer() {
        throw new UnsupportedOperationException();
    }
    default StopWatch getTrainEstimateTimer() {
        return new StopWatch();
    }

    default long getTrainTimeNanos() {
        return getTrainTimer().getTimeNanos();
    }

    default long getTrainEstimateTimeNanos() {
        return getTrainEstimateTimer().getTimeNanos();
    }
}
