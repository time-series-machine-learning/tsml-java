
package evaluation.evaluators;

import evaluation.evaluators.Evaluator;
import evaluation.storage.ClassifierResults;
import java.util.concurrent.TimeUnit;
import static utilities.GenericTools.indexOfMax;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Simply gathers predictions from the (assumed to have already been built/trained) passed classifier
 * on the data given, assumed to have already been sampled/set up as desired. 
 * 
 * distributionForInstance(Instance) MUST be defined, even if the classifier only really returns 
 * a one-hot distribution
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class SingleTestSetEvaluator extends Evaluator {
    
    public SingleTestSetEvaluator() {
        super(0,false,false);
    }
    public SingleTestSetEvaluator(int seed, boolean cloneData, boolean setClassMissing) {
        super(seed,cloneData,setClassMissing);
    }
    
    @Override
    public ClassifierResults evaluate(Classifier classifier, Instances dataset) throws Exception {
        Instances insts = null;
        if (cloneData)
            insts = new Instances(dataset);
        else 
            insts = dataset;
        
        ClassifierResults res = new ClassifierResults(insts.numClasses());
        res.setTimeUnit(TimeUnit.NANOSECONDS);
        
        res.turnOffZeroTimingsErrors();
        for (Instance testinst : insts) {
            double trueClassVal = testinst.classValue();
            if (setClassMissing)
                testinst.setClassMissing();
            
            long startTime = System.nanoTime();
            double[] dist = classifier.distributionForInstance(testinst);
            long predTime = System.nanoTime() - startTime;
            res.addPrediction(trueClassVal, dist, indexOfMax(dist), predTime, "");
        }
        res.turnOnZeroTimingsErrors();
        
        res.findAllStatsOnce(); 
        return res;
    }

}
