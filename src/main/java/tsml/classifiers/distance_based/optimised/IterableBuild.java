package tsml.classifiers.distance_based.optimised;

import tsml.classifiers.distance_based.utils.classifiers.contracting.ProgressiveBuild;
import tsml.data_containers.TimeSeriesInstances;

public interface IterableBuild extends ProgressiveBuild {
    
    void beforeBuild() throws Exception;
    
    boolean hasNextBuildStep() throws Exception;
    
    void nextBuildStep() throws Exception;
    
    void afterBuild() throws Exception;

    @Override default boolean isFullyBuilt() {
        try {
            return !hasNextBuildStep();
        } catch(Exception e) {
            throw new IllegalStateException(e);
        }
    }
    
    void setTrainData(TimeSeriesInstances trainData);
    
    default void buildClassifier(TimeSeriesInstances trainData) throws Exception {
        
        beforeBuild();
        while(hasNextBuildStep()) {
            nextBuildStep();
        }
        afterBuild();
    }
}
