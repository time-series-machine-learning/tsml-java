package evaluation.evaluators;

import evaluation.storage.ClassifierResults;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 *
 * @author xmw13bzu
 */
public abstract class Evaluator {
    int seed;
    
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
    
    public Evaluator(int seed, boolean cloneData, boolean setClassMissing) {
        this.seed = seed;
        this.cloneData = cloneData;
        this.setClassMissing = setClassMissing;
    }
    
    public int getSeed() {
        return seed;
    }
    
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
    
    public abstract ClassifierResults evaluate(Classifier classifier, Instances dataset) throws Exception;
}
