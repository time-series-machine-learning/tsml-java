package tsml.classifiers.distance_based.distances;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import java.util.function.Function;
import tsml.filters.Derivative;
import weka.core.DistanceFunction;
import weka.core.Instance;

public class WDDTW extends ImmutableTransformedDistanceMeasure {


    public WDDTW() {
        super("WDDTW", Derivative.getGlobalCache(), new WDTWDistance());
    }
}
