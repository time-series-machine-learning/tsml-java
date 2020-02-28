package tsml.classifiers.distance_based.distances;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import tsml.classifiers.distance_based.distances.transformed.TransformedDistanceMeasure;
import tsml.filters.Derivative;

public class WDDTW extends TransformedDistanceMeasure {


    public WDDTW() {
        super("WDDTW", Derivative.getGlobalCache(), new WDTWDistance());
    }
}
