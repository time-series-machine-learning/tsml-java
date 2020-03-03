package tsml.classifiers;

import evaluation.storage.ClassifierResults;
import utilities.Copy;
import utilities.Debugable;
import utilities.Randomised;
import utilities.params.ParamHandler;

import java.io.Serializable;

public interface TrainEstimateable {
    ClassifierResults getTrainResults();
    boolean getEstimateOwnPerformance();
    void setEstimateOwnPerformance(boolean estimateOwnPerformance);
}
