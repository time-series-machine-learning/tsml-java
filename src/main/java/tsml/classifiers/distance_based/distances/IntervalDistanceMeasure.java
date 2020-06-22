package tsml.classifiers.distance_based.distances;

import org.junit.Assert;
import weka.core.Instance;

public class IntervalDistanceMeasure extends BaseDistanceMeasure {



    @Override protected double findDistance(final Instance a, final Instance b, final double limit) {
        return 0;
    }
}
