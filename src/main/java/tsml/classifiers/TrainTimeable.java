package tsml.classifiers;

public interface TrainTimeable {
    default long getTrainTimeNanos() { return -1; }
    default long getTrainEstimateTimeNanos() { return 0; }
    default long getTrainPlusEstimateTimeNanos() {
        return getTrainEstimateTimeNanos() + getTrainTimeNanos();
    }
}
