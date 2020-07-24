package tsml.classifiers.distance_based.utils.system.timing;

import tsml.classifiers.TrainTimeable;

public interface TimedTrain extends TrainTimeable {
    StopWatch getTrainTimer();

    default long getTrainTime() {
        return getTrainTimer().getTime();
    }
}
