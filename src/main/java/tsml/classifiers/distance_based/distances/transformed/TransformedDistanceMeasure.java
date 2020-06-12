package tsml.classifiers.distance_based.distances.transformed;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import org.junit.Assert;
import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.filters.CachedFilter;
import tsml.filters.HashFilter;
import tsml.filters.Utilities;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;
import tsml.transformers.Transformer;

public class TransformedDistanceMeasure extends BaseDistanceMeasure implements TransformedDistanceMeasureable {

    // todo get and set params

    public TransformedDistanceMeasure(String name, Transformer transformer,
        DistanceFunction distanceFunction) {
        setName(name);
        setDistanceFunction(distanceFunction);
        setTransformer(transformer);
    }

    protected TransformedDistanceMeasure() {

    }

    private DistanceFunction distanceFunction;
    private Transformer transformer;
    private String name = getClass().getSimpleName();

    protected void setName(String name) {
        Assert.assertNotNull(name);
        this.name = name;
    }

    @Override
    public void setInstances(Instances data) {
        super.setInstances(data);
        distanceFunction.setInstances(data);
    }

    @Override
    public String getName() {
        return name;
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    protected void setDistanceFunction(DistanceFunction distanceFunction) {
        Assert.assertNotNull(distanceFunction);
        this.distanceFunction = distanceFunction;
    }

    public Transformer getTransformer() {
        return transformer;
    }

    protected void setTransformer(Filter transformer) {
        Assert.assertNotNull(transformer);
        this.transformer = transformer;
    }

    @Override
    public double findDistance(final Instance a, final Instance b, final double limit) {
        try {
            final Instance firstTransformed = transformer.transform(first);
            final Instance secondTransformed = transformer.transform(second);
            return distanceFunction.distance(firstTransformed, secondTransformed, cutOffValue, stats);
        } catch(Exception e) {
            throw new IllegalStateException(e);
        }
    }

    @Override public ParamSet getParams() {
        return super.getParams().add(TRANSFORMER_FLAG, transformer).add(DistanceMeasureable.DISTANCE_MEASURE_FLAG,
                                                                  distanceFunction);
    }

    public static final String TRANSFORMER_FLAG = "t";

    @Override
    public void setTraining(final boolean training) {
        super.setTraining(training);
        if(transformer instanceof CachedFilter) {
            if(((CachedFilter) transformer).getCache().isRead()) {
                // disable the cache writing if not in training mode
                // don't want to keep instances / transformation from the test data as the cache will explode along with
                // your computer
                ((CachedFilter) transformer).getCache().setWrite(training);
            }
        }
    }

}
