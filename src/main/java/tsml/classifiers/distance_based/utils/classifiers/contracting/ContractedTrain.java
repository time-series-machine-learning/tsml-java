package tsml.classifiers.distance_based.utils.classifiers.contracting;

import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.TrainTimeable;

public interface ContractedTrain extends TrainTimeContractable, TimedTrain, ProgressiveBuild {

    /**
     * Is the classifier fully built? This is irrelevant of contract timings and is instead a reflection of whether
     * work remains and further time could be allocated to the classifier to build the model further.
     * @return
     */
    @Override boolean isFullyBuilt();

    long getTrainTimeLimit();

    default boolean hasTrainTimeLimit() {
        return getTrainTimeLimit() > 0;
    }

    /**
     * 
     * @param nanos the amount of time currently taken (or the expectation of how long something will take and thus
     *              whether there is enough time to complete it). E.g. this could be the current run time of a clsf plus
     *              the predicted time to do some unit of work to improve the classifier. The result would indicate if
     *              there's enough time to get this unit of work done within the contract and can therefore be used to
     *              decide whether to do it in the first place.
     * @return
     */
    default boolean insideTrainTimeLimit(long nanos) {
        return !hasTrainTimeLimit() || nanos < getTrainTimeLimit();
    }

    default long findRemainingTrainTime(long trainTime) {
        if(!hasTrainTimeLimit()) {
            return Long.MAX_VALUE;
        }
        final long trainTimeLimit = getTrainTimeLimit();
        return trainTimeLimit - trainTime;
    }
    
    default long findRemainingTrainTime() {
        return findRemainingTrainTime(getRunTime());
    }
}
