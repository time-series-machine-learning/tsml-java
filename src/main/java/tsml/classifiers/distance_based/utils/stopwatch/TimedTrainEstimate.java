package tsml.classifiers.distance_based.utils.stopwatch;

public interface TimedTrainEstimate extends TrainEstimateTimeable {
    StopWatch getTrainEstimateTimer();

    default long getTrainEstimateTime() {
        return getTrainEstimateTimer().getTime();
    }
}
