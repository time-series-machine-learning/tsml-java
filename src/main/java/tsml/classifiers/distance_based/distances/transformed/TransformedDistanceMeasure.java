package tsml.classifiers.distance_based.distances.transformed;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.filters.CachedFilter;
import tsml.filters.Utilities;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;
import weka.filters.Filter;

public class TransformedDistanceMeasure extends BaseDistanceMeasure implements TransformedDistanceMeasureable {

    // todo get and set params

    public TransformedDistanceMeasure(String name, Filter transformer,
        DistanceFunction distanceFunction) {
        setName(name);
        setDistanceFunction(distanceFunction);
        setTransformer(transformer);
    }

    protected TransformedDistanceMeasure() {

    }

    private DistanceFunction distanceFunction;
    private Filter transformer;
    private String name = getClass().getSimpleName();

    protected void setName(String name) {
        if(name == null) throw new NullPointerException();
        this.name = name;
    }

    @Override
    public void setInstances(Instances data) {
        super.setInstances(data);
        distanceFunction.setInstances(data);
        try {
            transformer.setInputFormat(data);
        } catch(Exception e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public String getName() {
        return name;
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    protected void setDistanceFunction(DistanceFunction distanceFunction) {
        if(distanceFunction == null) throw new NullPointerException();
        this.distanceFunction = distanceFunction;
    }

    public Filter getTransformer() {
        return transformer;
    }

    protected void setTransformer(Filter transformer) {
        if(transformer == null) throw new NullPointerException();
        this.transformer = transformer;
    }

    @Override
    public double distance(final Instance first, final Instance second, final double cutOffValue,
                           final PerformanceStats stats) {
        try {
            final Instance firstTransformed = Utilities.filter(first, transformer);
            final Instance secondTransformed = Utilities.filter(second, transformer);
            return distanceFunction.distance(firstTransformed, secondTransformed, cutOffValue, stats);
        } catch(Exception e) {
            throw new IllegalStateException(e);
        }
    }

    @Override public ParamSet getParams() {
        return super.getParams().add(getTransformerFlag(), getTransformer()).add(DistanceMeasureable.getDistanceFunctionFlag(),
                                                                  getDistanceFunction());
    }

    public static String getTransformerFlag() {
        return "t";
    }

    @Override
    public void setTraining(final boolean training) {
        super.setTraining(training);
        if(transformer instanceof CachedFilter) {
            // always enable cache read, doesn't matter if the cache is empty
            ((CachedFilter) transformer).getCache().setRead(true);
            // disable the cache writing if not in training mode
            // don't want to keep instances / transformation from the test data as the cache will explode along with
            // your computer
            ((CachedFilter) transformer).getCache().setWrite(training);
        }
    }

}
