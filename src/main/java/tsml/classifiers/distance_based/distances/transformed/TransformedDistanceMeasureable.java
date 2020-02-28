package tsml.classifiers.distance_based.distances.transformed;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import weka.core.DistanceFunction;
import weka.filters.Filter;

public interface TransformedDistanceMeasureable extends DistanceMeasureable {
    DistanceFunction getDistanceFunction();
    Filter getTransformer();
    static String getTransformerFlag() {
        return "f";
    }
}
