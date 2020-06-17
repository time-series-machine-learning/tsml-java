package tsml.classifiers.distance_based.proximity;

import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;

public class CrossValidation extends BaseTrainEstimateMethod {

    private int numFolds;
    public final static String NUM_FOLDS_FLAG = "f";

    public CrossValidation(int numFolds) {
        setNumFolds(numFolds);
    }

    public int getNumFolds() {
        return numFolds;
    }

    public void setNumFolds(final int numFolds) {
        this.numFolds = numFolds;
    }

    @Override public ParamSet getParams() {
        return super.getParams().add(NUM_FOLDS_FLAG, numFolds);
    }

    @Override public void setParams(final ParamSet paramSet) {
        super.setParams(paramSet);
        ParamHandler.setParam(paramSet, NUM_FOLDS_FLAG, this::setNumFolds, Integer.class);
    }
}
