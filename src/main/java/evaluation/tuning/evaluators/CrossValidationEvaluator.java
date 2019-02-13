
package evaluation.tuning.evaluators;

import evaluation.ClassifierResults;
import evaluation.CrossValidator;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class CrossValidationEvaluator implements Evaluator {
    int numCVFolds;

    int seed = 0;
    
    public CrossValidationEvaluator() {
        this.numCVFolds = 10;
    }
    
    public int getNumCVFolds() {
        return numCVFolds;
    }

    public void setNumCVFolds(int numCVFolds) {
        this.numCVFolds = numCVFolds;
    }
    
    private void checkNumCVFolds(Instances dataset) { 
        if (dataset.numInstances() < numCVFolds)
            numCVFolds = dataset.numInstances();
    }
    
    @Override
    public ClassifierResults evaluate(Classifier classifier, Instances dataset) throws Exception {
        checkNumCVFolds(dataset);
        
        CrossValidator cv = new CrossValidator();
        cv = new evaluation.CrossValidator();
        cv.setSeed(seed);
        cv.setNumFolds(numCVFolds);
        cv.buildFolds(dataset);
        
        
        ClassifierResults res = cv.crossValidateWithStats(classifier, dataset);
        res.findAllStatsOnce();
        return res;
    }

    @Override
    public void setSeed(int seed) {
        this.seed = seed;
    }

}
