package tsml.classifiers.distance_based.distances.transformed;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import tsml.transformers.Transformer;
import weka.core.DistanceFunction;

public interface TransformDistanceMeasureable extends TransformedDistanceMeasureable {
    void setTransformer(Transformer transformer);
    void setDistanceFunction(DistanceFunction distanceFunction);
}
