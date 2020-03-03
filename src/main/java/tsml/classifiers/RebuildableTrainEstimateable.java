package tsml.classifiers;

public interface RebuildableTrainEstimateable extends Rebuildable, TrainEstimateable {

    boolean isRegenerateTrainEstimate();

    void setRegenerateTrainEstimate(boolean regenerateTrainEstimate);
}
