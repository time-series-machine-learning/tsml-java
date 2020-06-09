package tsml.classifiers.distance_based.distances;

import org.junit.Assert;
import tsml.classifiers.distance_based.utils.instance.ExposedDenseInstance;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.classifiers.distance_based.utils.strings.StrUtils;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;

/**
 * Abstract distance measure. This takes the weka interface for DistanceFunction and implements some default methods,
 * adding several checks and balances also. All distance measures should extends this class.
 * <p>
 * Contributors: goastler
 */
public abstract class BaseDistanceMeasure implements DistanceMeasureable {

    // check for whether setInstances has been called before doing any distance measurements
    private transient boolean dataHasBeenSet = false;
    // the data which was passed to setInstances
    private transient Instances data;
    // is this in training phase or testing phase
    private transient boolean training = true;
    // whether to set the first instance to the longest instance when calling the distance function
    private transient boolean longestInstanceFirst = true;


    // optional check for data in the correct format
    protected void checkData(Instance a, Instance b) {
        Assert.assertEquals(a.numAttributes() - 1, a.classIndex());
        Assert.assertEquals(b.numAttributes() - 1, b.classIndex());
//        if(!dataHasBeenSet) {
//            throw new IllegalStateException("must call setInstances first to setup the distance measure");
//        } todo do we need this? some dm's might need to look at the instances first perhaps
    }

    @Override
    public double getMaxDistance() {
        return Double.POSITIVE_INFINITY;
    }

    public boolean isSymmetric() {
        return true;
    }

    @Override
    public final double distance(final Instance a, final Instance b) {
        return distance(a, b, getMaxDistance());
    }

    @Override
    public final double distance(Instance a, Instance b, final double limit) {
        checkData(a, b);
        // put a or first as the longest time series
        if(longestInstanceFirst && b.numAttributes() > a.numAttributes()) {
            Instance tmp = a;
            a = b;
            b = tmp;
        }
        return findDistance(a, b, limit);
    }

    protected abstract double findDistance(final Instance a, final Instance b, final double limit);

    @Override
    public final double distance(final Instance a, final Instance b, final double limit,
        final PerformanceStats stats) {
        return distance(a, b, limit);
    }

    @Override
    public void postProcessDistances(final double[] distances) {

    }

    @Override
    public String getName() {
        return getClass().getSimpleName();
    }

    @Override
    public final double distance(final Instance a, final Instance b, final PerformanceStats stats) throws Exception {
        return distance(a, b, getMaxDistance(), stats);
    }

    @Override
    public String toString() {
        return getName() + " " + StrUtils.join(" ", getOptions());
    }

    @Override
    public Instances getInstances() {
        return data;
    }

    @Override
    public void setAttributeIndices(final String value) {

    }

    @Override
    public String getAttributeIndices() {
        return null;
    }

    @Override
    public void setInvertSelection(final boolean value) {

    }

    @Override
    public boolean getInvertSelection() {
        return false;
    }

    @Override
    public void setInstances(final Instances data) {
        this.data = data;
        dataHasBeenSet = true;
        if(data != null) {
            if(data.classIndex() != data.numAttributes() - 1) {
                throw new IllegalStateException("class value must be at the end");
            }
        }
    }

    public void clean() {
        data = null;
        dataHasBeenSet = false;
    }

    @Override
    public void update(final Instance ins) {
        data.add(ins);
    }

    @Override public void setParams(final ParamSet param) {

    }

    @Override public ParamSet getParams() {
        return new ParamSet();
    }

    @Override
    public boolean isTraining() {
        return training;
    }

    @Override
    public void setTraining(final boolean training) {
        this.training = training;
    }

    public boolean isLongestInstanceFirst() {
        return longestInstanceFirst;
    }

    protected void setLongestInstanceFirst(final boolean longestInstanceFirst) {
        this.longestInstanceFirst = longestInstanceFirst;
    }
}
