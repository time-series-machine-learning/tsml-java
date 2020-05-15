package tsml.classifiers.distance_based.utils.stopwatch;

import tsml.classifiers.TrainTimeable;

public interface TimedTrain extends TrainTimeable {
    StopWatch getTrainTimer();

    default long getTrainTime() {
        return getTrainTimer().getTimeNanos();
    }
}
