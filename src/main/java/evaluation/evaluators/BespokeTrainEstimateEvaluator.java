package evaluation.evaluators;

import evaluation.storage.ClassifierResults;
import timeseriesweka.classifiers.TrainAccuracyEstimator;
import weka.classifiers.Classifier;
import weka.core.Instances;

public class BespokeTrainEstimateEvaluator extends Evaluator {

    @Override
    public ClassifierResults evaluate(final Classifier classifier, final Instances dataset) throws
                                                                                            Exception {
        if(classifier instanceof TrainAccuracyEstimator) {
            ((TrainAccuracyEstimator) classifier).setFindTrainAccuracyEstimate(true);
            classifier.buildClassifier(dataset);
            return ((TrainAccuracyEstimator) classifier).getTrainResults();
        }
        throw new UnsupportedOperationException();
    }
}
