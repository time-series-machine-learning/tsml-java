package tsml.classifiers.distance_based.distances.transformed;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
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
    private String name;

    protected void setName(String name) {
        if(name == null) throw new NullPointerException();
        this.name = name;
    }

    @Override
    public void setInstances(Instances data) {
        super.setInstances(data);
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
    public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats) {
        return distanceFunction.distance(first, second, cutOffValue, stats);
    }
}
