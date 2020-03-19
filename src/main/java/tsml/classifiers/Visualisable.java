package tsml.classifiers;

public interface Visualisable {
    default long getTrainTimeNanos() { return -1; }
}
