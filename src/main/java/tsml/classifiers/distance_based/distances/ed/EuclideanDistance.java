package tsml.classifiers.distance_based.distances.ed;

import tsml.classifiers.distance_based.distances.ArrayBasedDistanceMeasure;
import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.utils.instance.ExposedDenseInstance;
import utilities.Utilities;
import weka.core.Instance;

public class EuclideanDistance extends ArrayBasedDistanceMeasure {

    @Override
    public double findDistance(final double[] a, double[] b, final double limit) {
        double sum = 0;

        int aLength = a.length - 1;
        int bLength = b.length - 1;

        for(int i = 0; i < aLength; i++) {
            sum += Math.pow(a[i] - b[i], 2);
            if(sum > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }

        return sum;
    }
}
