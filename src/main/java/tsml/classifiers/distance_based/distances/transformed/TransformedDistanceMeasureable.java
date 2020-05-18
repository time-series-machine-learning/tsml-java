package tsml.classifiers.distance_based.distances.transformed;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.transformers.Transformer;
import weka.core.DistanceFunction;

public interface TransformedDistanceMeasureable extends DistanceMeasureable {
    DistanceFunction getDistanceFunction();
    Transformer getTransformer();
    static String getTransformerFlag() {
        return "f";
    }
}
