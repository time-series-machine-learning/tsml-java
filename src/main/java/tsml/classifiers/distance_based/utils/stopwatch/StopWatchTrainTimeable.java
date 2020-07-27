package tsml.classifiers.distance_based.utils.stopwatch;

import tsml.classifiers.TrainTimeable;

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

    default long getTrainTime() {
        return getTrainTimer().getTimeNanos();
    }

    default long getTrainEstimateTimeNanos() {
        return getTrainEstimateTimer().getTimeNanos();
    }
}
