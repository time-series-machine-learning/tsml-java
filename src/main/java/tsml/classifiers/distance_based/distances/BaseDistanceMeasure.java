package tsml.classifiers.distance_based.distances;

import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.strings.StrUtils;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;

/**
 * Abstract distance measure. This takes the weka interface for DistanceFunction and implements some default methods,
 * adding several checks and balances also. All distance measures should extends this class. This is loosely based on
 * the Transformer pattern whereby the user optionally "fits" some data and can then proceed to use the distance
 * measure. Simple distance measures need not fit at all, therefore the fit method is empty for those implementations
 * . fit() should always be called before any distance measurements.
 * <p>
 * Contributors: goastler
 */
public abstract class BaseDistanceMeasure implements DistanceMeasure {

    public BaseDistanceMeasure() {
        setName(getClass().getSimpleName());
        setLongestInstanceFirst(true);
        setStoreData(false);
        setInvertSelection(false);
    }

    // check for whether setInstances has been called before doing any distance measurements
    private boolean fitted;
    // whether to set the first instance to the longest instance when calling the distance function
    private boolean longestInstanceFirst;
    // the name of this distance measure
    private String name;
    // whether to store the fitted data
    private boolean storeData;
    // the stored data
    private Instances data;
    // whether to invert the distance measure. false --> the smaller the distance the more similar. true --> the
    // larger the distance the more similar
    private boolean invert;

    @Override
    public String toString() {
        return getName() + " " + StrUtils.join(" ", getOptions());
    }

    /**
     * clean up any data collected during fitting
     */
    public void clean() {
        fitted = false;
        data = null;
    }

    public boolean isLongestInstanceFirst() {
        return longestInstanceFirst;
    }

    protected void setLongestInstanceFirst(final boolean longestInstanceFirst) {
        this.longestInstanceFirst = longestInstanceFirst;
    }    @Override
    public double getMaxDistance() {
        return Double.POSITIVE_INFINITY;
    }

    /**
     * whether the fit method has been called
     *
     * @return
     */
    public boolean isFitted() {
        return fitted;
    }

    private void setFitted(final boolean fitted) {
        this.fitted = fitted;
    }    /**
     * whether the distance measure is symmetric, i.e. dist(a,b) == dist(b,a)
     *
     * @return
     */
    public boolean isSymmetric() {
        return true;
    }

    @Override public Instances getInstances() {
        return data;
    }

    @Override
    public final double distance(final Instance a, final Instance b) {
        return distance(a, b, getMaxDistance());
    }

    @Override
    public final double distance(Instance a, Instance b, final double limit) {
        // make sure class labels are at the end
        Assert.assertEquals(a.numAttributes() - 1, a.classIndex());
        Assert.assertEquals(b.numAttributes() - 1, b.classIndex());
        // put a as the longest time series
        if(longestInstanceFirst && b.numAttributes() > a.numAttributes()) {
            Instance tmp = a;
            a = b;
            b = tmp;
        }
        return findDistance(a, b, limit);
    }

    protected abstract double findDistance(final Instance a, final Instance b, final double limit);

    @Override
    public final double distance(final Instance a, final Instance b, final double limit, final PerformanceStats stats) {
        return distance(a, b, limit);
    }

    @Override
    public void postProcessDistances(final double[] distances) {

    }

    @Override
    public String getName() {
        return name;
    }

    @Override public void setName(final String name) {
        Assert.assertNotNull(name);
        this.name = name;
    }

    @Override
    public final double distance(final Instance a, final Instance b, final PerformanceStats stats) throws Exception {
        return distance(a, b, getMaxDistance(), stats);
    }

    @Override
    public void setAttributeIndices(final String value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public String getAttributeIndices() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setInvertSelection(final boolean value) {
        invert = value;
    }

    @Override
    public boolean getInvertSelection() {
        return invert;
    }

    /**
     * fit the distance measure to this data. This is the equivalent of fit()
     *
     * @param data
     */
    public void setInstances(Instances data) {
        Assert.assertNotNull(data);
        if(storeData) {
            this.data = data;
        }
        fitted = true;
        if(data.classIndex() != data.numAttributes() - 1) {
            throw new IllegalStateException("class value must be at the end");
        }
    }

    @Override
    public void update(final Instance ins) {
        if(!fitted) {
            throw new IllegalStateException("not fitted");
        }
        if(storeData) {
            data.add(ins);
        }
    }

    @Override public void setParams(final ParamSet param) throws Exception {

    }

    @Override public ParamSet getParams() {
        return new ParamSet();
    }

    public boolean isStoreData() {
        return storeData;
    }

    public void setStoreData(final boolean storeData) {
        this.storeData = storeData;
    }

}
