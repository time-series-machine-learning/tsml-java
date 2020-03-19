package tsml.classifiers;

public interface Interperetable {
    default long getTrainTimeNanos() { return -1; }
}
