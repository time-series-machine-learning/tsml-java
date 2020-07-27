package tsml.classifiers.distance_based.distances.ed;

import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import weka.core.Instance;

public class EDistance extends BaseDistanceMeasure {

    public double findDistance(final Instance a, Instance b, final double limit) {
        double sum = 0;

        int aLength = a.numAttributes() - 1;
        int bLength = b.numAttributes() - 1;

        for(int i = 0; i < aLength; i++) {
            sum += Math.pow(a.value(i) - b.value(i), 2);
            if(sum > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }

        return sum;
    }
}
