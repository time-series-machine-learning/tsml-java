package tsml.classifiers.distance_based.distances;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import java.util.function.Function;
import weka.classifiers.evaluation.output.prediction.Null;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.neighboursearch.PerformanceStats;

public class TransformedDistanceMeasure extends ImmutableTransformedDistanceMeasure {

    public TransformedDistanceMeasure(String name,
        Function<Instance, Instance> transformer, DistanceFunction distanceFunction) {
        super(name, transformer, distanceFunction);
    }

    @Override
    public void setDistanceFunction(DistanceFunction distanceFunction) {
        super.setDistanceFunction(distanceFunction);
    }

    @Override
    public void setTransformer(Function<Instance, Instance> transformer) {
        super.setTransformer(transformer);
    }

    @Override
    public void setName(String name) {
        super.setName(name);
    }
}
