package tsml.classifiers.distance_based.knn;

import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.classifiers.distance_based.utils.params.ParamUtils;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class K implements ParamHandler {

    private int k = 1;
    public static final String K_FLAG = "k";

    public K() {

    }

    @Override
    public ParamSet getParams() {
        return new ParamSet().add(K_FLAG, k);
    }

    public K setK(final int k) {
        this.k = k;
        return this;
    }

    @Override
    public void setParams(final ParamSet paramSet) {
        ParamUtils.setSingleValuedParam(paramSet, K_FLAG, this::setK, Integer::valueOf);
    }

    public static void main(String[] args) {
        K k = new K();
        k.setParams(new ParamSet().add(K_FLAG, "3"));
        System.out.println();
    }
}
