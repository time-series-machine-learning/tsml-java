package tsml.classifiers;

public interface TrainTimeable {
    default long getTrainTimeNanos() { return -1; };
}
