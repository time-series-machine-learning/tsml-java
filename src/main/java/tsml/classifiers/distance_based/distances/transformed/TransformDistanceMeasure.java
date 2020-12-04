package tsml.classifiers.distance_based.distances.transformed;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.transformers.Transformer;
import weka.core.DistanceFunction;

public interface TransformDistanceMeasure extends DistanceMeasure {
    DistanceMeasure getDistanceMeasure();
    Transformer getTransformer();
    void setDistanceMeasure(DistanceMeasure distanceMeasure);
    void setTransformer(Transformer transformer);
}
