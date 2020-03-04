package tsml.classifiers;

import tsml.classifiers.distance_based.utils.StopWatch;

/**
 * Purpose: simple interface to defer TrainTimeable to StopWatch implementations.
 *
 * Contributors: goastler
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
