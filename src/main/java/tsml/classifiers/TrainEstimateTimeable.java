package tsml.classifiers;

public interface TrainEstimateTimeable extends TrainTimeable {

    long getTrainEstimateTimeNanos();

    default long getTrainPlusEstimateTimeNanos() {
        return getTrainEstimateTimeNanos() + getTrainTimeNanos();
    }

}
