package tsml.classifiers.distance_based.utils.contracting;

import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.TrainTimeable;

public interface ContractedTrain extends TrainTimeContractable, TrainTimeable {

    long getTrainTimeLimit();

    default boolean hasTrainTimeLimit() {
        return getTrainTimeLimit() > 0;
    }

    default boolean insideTrainTimeLimit(long nanos) {
        return !hasTrainTimeLimit() || nanos < getTrainTimeLimit();
    }
}
