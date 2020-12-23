package tsml.classifiers.distance_based.distances;

import tsml.classifiers.distance_based.utils.strings.StrUtils;
import weka.core.Utils;

public abstract class BaseDistanceMeasure implements DistanceMeasure {

    @Override public String toString() {
        final String str = Utils.joinOptions(getOptions());
        if(str.isEmpty()) {
            return getName();
        } else {
            return getName() + " " + str;
        }
    }
}
