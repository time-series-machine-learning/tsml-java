package tsml.classifiers;

public interface TestTimeable {
    default long getTestTimeNanos() { return -1; };
}
