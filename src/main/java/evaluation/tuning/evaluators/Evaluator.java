package evaluation.tuning.evaluators;

import evaluation.ClassifierResults;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 *
 * @author xmw13bzu
 */
public interface Evaluator {
    public void setSeed(int seed);
    public ClassifierResults evaluate(Classifier classifier, Instances dataset) throws Exception;
}
