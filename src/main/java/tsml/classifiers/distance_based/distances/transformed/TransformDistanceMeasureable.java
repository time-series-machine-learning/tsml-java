package tsml.classifiers.distance_based.distances.transformed;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import weka.core.DistanceFunction;
import weka.filters.Filter;

public interface TransformDistanceMeasureable extends TransformedDistanceMeasureable {
    void setTransformer(Filter transformer);
    void setDistanceFunction(DistanceFunction distanceFunction);
}
