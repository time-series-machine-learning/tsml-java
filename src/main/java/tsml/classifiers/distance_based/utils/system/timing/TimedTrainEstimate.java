package tsml.classifiers.distance_based.utils.system.timing;

public interface TimedTrainEstimate extends TrainEstimateTimeable {
    StopWatch getTrainEstimateTimer();

    default long getTrainEstimateTimeNanos() {
        return getTrainEstimateTimer().getTime();
    }
}
