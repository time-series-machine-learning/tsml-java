package tsml.classifiers.distance_based.distances;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import java.util.function.Function;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.neighboursearch.PerformanceStats;

public class TransformedDistanceMeasure extends AbstractDistanceMeasure {

    public TransformedDistanceMeasure(String name, Function<Instance, Instance> transformer, DistanceFunction distanceFunction) {
        super(name);
        setDistanceFunction(distanceFunction);
        setTransformer(transformer);
    }

    private DistanceFunction distanceFunction;
    private Function<Instance, Instance> transformer;

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    public void setDistanceFunction(DistanceFunction distanceFunction) {
        if(distanceFunction == null) throw new NullPointerException();
        this.distanceFunction = distanceFunction;
    }

    public Function<Instance, Instance> getTransformer() {
        return transformer;
    }

    public void setTransformer(Function<Instance, Instance> transformer) {
        if(transformer == null) throw new NullPointerException();
        this.transformer = transformer;
    }

    @Override
    public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats) {
        return distanceFunction.distance(first, second, cutOffValue, stats);
    }
}
