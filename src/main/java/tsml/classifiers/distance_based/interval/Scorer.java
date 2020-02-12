package tsml.classifiers.distance_based.interval;

public interface Scorer {
    double findScore(int parent, int... children);
}
