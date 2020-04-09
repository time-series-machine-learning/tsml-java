package tsml.classifiers.distance_based.proximity.splitting.exemplar_based;

import java.util.List;
import weka.core.DistanceFunction;

/**
 * Purpose: pick a distance function.
 * <p>
 * Contributors: goastler
 */
public interface DistanceFunctionPicker {

    DistanceFunction pickDistanceFunction();
}
