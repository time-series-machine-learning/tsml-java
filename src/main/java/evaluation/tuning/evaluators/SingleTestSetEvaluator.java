
package evaluation.tuning.evaluators;

import evaluation.ClassifierResults;
import static utilities.GenericTools.indexOfMax;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Simply gathers predictions from the (assumed to have already been built/trained) passed classifier
 * on the data given, assumed to have already been sampled/set up as desired. 
 * 
 * distributionForInstance(Instance) must be defined, even if the classifier only really returns 
 * a one-hot distribution
 * 
 * Will *NOT* call setClassMissing() on each instance,
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class SingleTestSetEvaluator implements Evaluator {
    
    /**
     * Flag for whether to clone the data. Defaults to false, as no classifier should 
     * be editing the data itself when training/testing, however setting this to true 
     * will guarantee that the same (jave) instantiations of (weka) instance(s) objects
     * can be reused in higher-level experimental code. 
     */
    boolean cloneData;
    
    /**
     * Each instance will have setClassMissing() called upon it. To ABSOLUTELY enforce that 
     * no classifier can cheat in any way (e.g some filter/transform inadvertently incorporates the class 
     * value back into the transformed data set). 
     * 
     * The only reason to leave this as false (as it has been by default, for backwards compatability reasons)
     * is that in higher level experimental code, the same (jave) instantiations of (weka) instance(s) objects are used multiple 
     * times, and the latter expects the class value to still be there (to check for correct predictions, e.g)
     */
    boolean setClassMissing;

   
    int seed = 0;
    
    public SingleTestSetEvaluator() {
        this.cloneData = false;
        this.setClassMissing = false;
    }
    public SingleTestSetEvaluator(boolean cloneData, boolean setClassMissing) {
        this.cloneData = cloneData;
        this.setClassMissing = setClassMissing;
    }
    
    @Override
    public void setSeed(int seed) {
        this.seed = seed;
    }

    public boolean getCloneData() {
        return cloneData;
    }

    public void setCloneData(boolean cloneData) {
        this.cloneData = cloneData;
    }

    public boolean getSetClassMissing() {
        return setClassMissing;
    }

    public void setSetClassMissing(boolean setClassMissing) {
        this.setClassMissing = setClassMissing;
    }
    
    @Override
    public ClassifierResults evaluate(Classifier classifier, Instances dataset) throws Exception {
        Instances insts = null;
        if (cloneData)
            insts = new Instances(dataset);
        else 
            insts = dataset;
        
        ClassifierResults res = new ClassifierResults(insts.numClasses());
            
        for (Instance testinst : insts) {
            double trueClassVal = testinst.classValue();
            if (setClassMissing)
                testinst.setClassMissing();
            
            long startTime = System.currentTimeMillis();
            double[] dist = classifier.distributionForInstance(testinst);
            long predTime = System.currentTimeMillis() - startTime;
            res.addPrediction(trueClassVal, dist, indexOfMax(dist), predTime, "");
        }

        res.findAllStatsOnce(); 
        return res;
    }

}
