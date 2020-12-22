package tsml.classifiers.distance_based.distances.ed;

import tsml.classifiers.distance_based.distances.MatrixBasedDistanceMeasure;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.data_containers.TimeSeriesInstance;

public class EDistance extends MatrixBasedDistanceMeasure {
    
    public double distance(final TimeSeriesInstance a, TimeSeriesInstance b, final double limit) {
        double sum = 0;

        final int aLength = a.getMaxLength();
        final int bLength = b.getMaxLength();

        for(int i = 0; i < aLength; i++) {
            sum += DTWDistance.cost(a, i, b, i);
            if(sum > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }

        return sum;
    }
}
