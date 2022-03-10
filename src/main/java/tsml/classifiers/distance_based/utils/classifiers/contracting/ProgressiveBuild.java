package tsml.classifiers.distance_based.utils.classifiers.contracting;

public interface ProgressiveBuild {

    /**
     * Has the classifier been fully built yet?
     * @return true == yes, no further work can be done. false == no, more work can be done.
     */
    boolean isFullyBuilt();
}
