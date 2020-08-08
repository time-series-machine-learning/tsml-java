package tsml.classifiers;

/**
 * Purpose: track the time associated with producing a train estimate.
 *
 * Contributors: goastler
 */
public interface TrainEstimateTimeable extends TrainTimeable {

    long getTrainEstimateTime();

    default long getTrainPlusEstimateTime() {
        return getTrainEstimateTime() + getTrainTime();
    }

}
