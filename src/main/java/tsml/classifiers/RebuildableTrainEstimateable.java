package tsml.classifiers;

/**
 * Purpose: similar to Rebuildable, but controls the train estimate. If something changes in the classifier, say
 * contract time, and further work is done, the train estimate will likely need updating. This interface controls
 * that. This is important for classifiers where it's expensive to produce a train estimate and it is more beneficial
 * to manually control when the train estimate is regenerated.
 *
 * Contributors: goastler
 */
public interface RebuildableTrainEstimateable extends Rebuildable, TrainEstimateable {

    boolean isRegenerateTrainEstimate();

    void setRegenerateTrainEstimate(boolean regenerateTrainEstimate);
}
