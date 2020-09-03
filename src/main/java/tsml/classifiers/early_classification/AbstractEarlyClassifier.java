package tsml.classifiers.early_classification;

import weka.classifiers.AbstractClassifier;

public abstract class AbstractEarlyClassifier extends AbstractClassifier {

    protected int[] thresholds;

    protected boolean normalise = false;

    public int[] getThresholds(){
        return thresholds;
    }

    public void setThresholds(int[] t){
        thresholds = t;
    }

    public void setNormalise(boolean b) { normalise = b; }
}
