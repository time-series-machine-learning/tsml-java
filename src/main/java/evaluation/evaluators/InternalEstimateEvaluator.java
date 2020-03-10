package evaluation.evaluators;

import evaluation.storage.ClassifierResults;
import tsml.classifiers.EnhancedAbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * A dummy/wrapper evaluator for gathering the internal estimates of classifiers that satisfy
 * EnhancedAbstractClassifier.classifierAbleToEstimateOwnPerformance(classifier))
 *
 * Builds the classifier on the data, and returns the ClassifierResults object that is build internally by the classifier
 *
 * Currently, no additional meta info is supplied/forced into the results object by the evaluator, and so it is up to
 * experimenters and classifier authors to populate any additional meta-info needed/wanted
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class InternalEstimateEvaluator extends Evaluator {

    public InternalEstimateEvaluator() {
        super(0,false,false);
    }
    public InternalEstimateEvaluator(int seed, boolean cloneData, boolean setClassMissing) {
        super(seed,cloneData,setClassMissing);
    }

    @Override
    public synchronized ClassifierResults evaluate(Classifier classifier, Instances dataset) throws Exception {

        if (!EnhancedAbstractClassifier.classifierAbleToEstimateOwnPerformance(classifier))
            throw new IllegalArgumentException("To generate an internal estimate of performance, a classifier must extend " +
                    "EnhancedAbstractClassifier and have CAN_ESTIMATE_OWN_PERFORMANCE=true. Classifier class passed: " + classifier.getClass().getSimpleName());

        final Instances insts = cloneData ? new Instances(dataset) : dataset;

        EnhancedAbstractClassifier eac = (EnhancedAbstractClassifier) classifier;
        eac.setEstimateOwnPerformance(true);
        eac.setSeed(seed);
        eac.buildClassifier(insts);

        ClassifierResults res = eac.getTrainResults();
        res.findAllStatsOnce();

        return res;
    }

    @Override
    public Evaluator cloneEvaluator() {
        return new InternalEstimateEvaluator(this.seed, this.cloneData, this.setClassMissing);
    }
}
