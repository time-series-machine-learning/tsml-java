package experiments;

import tsml.classifiers.EnhancedAbstractClassifier;
import weka.core.Instances;

/**
 * Class for conducting experiments on a single problem
 * Wish list
 * 1. Cross validation: create a single output file with the cross validation predictions
 * 2. AUROC: take cross validation results and form the data for a AUROC plot
 * 3. Tuning: tune classifier on a train split
 * 4. Sensitivity: plot parameter space
 * 5. Robustness: performance with changing train set size (including acc estimates)
 */
public class SingleProblemExperiments {
    /**
     * Input, classifier, train set, test set, number of intervals (k)
     *
     * Train set will be resampled for k different train sizes at equally spaced intervals

     * Output to file results: TrainSize, TestAccActual, (TestAccEstimated, optional)
      */
    public static void increasingTrainSetSize(EnhancedAbstractClassifier c, Instances train, Instances test, int nIntervals, String results){
    // Work out intervals
        int fullLength=train.numInstances();
        int interval = fullLength/(nIntervals-1);
    //

    }

}
