package tsml.classifiers;

public interface Parallelisable extends Trainable {
    boolean isParallelisationEnabled();
    void setParallelisationEnabled(boolean state);
    boolean isFinalModel();
}
