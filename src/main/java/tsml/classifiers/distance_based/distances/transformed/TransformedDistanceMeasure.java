package tsml.classifiers.distance_based.distances.transformed;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.distances.wdtw.WDTW;
import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
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
        if(name == null) throw new NullPointerException();
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
        if(distanceFunction == null) throw new NullPointerException();
        this.distanceFunction = distanceFunction;
    }

    public Transformer getTransformer() {
        return transformer;
    }

    protected void setTransformer(Transformer transformer) {
        if(transformer == null) throw new NullPointerException();
        this.transformer = transformer;
    }

    @Override
    public double distance(final Instance first, final Instance second, final double cutOffValue,
                           final PerformanceStats stats) {
        try {
            final Instance firstTransformed = transformer.transform(first);
            final Instance secondTransformed = transformer.transform(second);
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
}
