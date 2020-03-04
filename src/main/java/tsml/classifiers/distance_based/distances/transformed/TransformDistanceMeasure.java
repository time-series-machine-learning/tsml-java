package tsml.classifiers.distance_based.distances.transformed;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import java.util.function.Function;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.filters.Filter;

public class TransformDistanceMeasure extends TransformedDistanceMeasure implements TransformDistanceMeasureable {

    public TransformDistanceMeasure(String name,
        Filter transformer, DistanceFunction distanceFunction) {
        super(name, transformer, distanceFunction);
    }

    @Override
    public void setDistanceFunction(DistanceFunction distanceFunction) {
        super.setDistanceFunction(distanceFunction);
    }

    @Override
    public void setTransformer(Filter transformer) {
        super.setTransformer(transformer);
    }

    @Override
    public void setName(String name) {
        super.setName(name);
    }



}
