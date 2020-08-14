package tsml.classifiers.early_classification;

import weka.core.Instances;

public interface LoadableEarlyDecisionMaker {

    void loadFromFile(Instances data, String directoryPath, int[] thresholds) throws Exception;
}
