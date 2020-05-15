package tsml.classifiers.distance_based.utils.contracting;

import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.TrainTimeable;
import tsml.classifiers.distance_based.utils.stopwatch.StopWatch;

public interface ContractedTrain extends TrainTimeContractable, TrainTimeable {

    long getTrainTimeLimit();

    default boolean hasTrainTimeLimit() {
        return getTrainTimeLimit() > 0;
    }

    default boolean insideTrainTimeLimit(long nanos) {
        return !hasTrainTimeLimit() || nanos < getTrainTimeLimit();
    }
}
