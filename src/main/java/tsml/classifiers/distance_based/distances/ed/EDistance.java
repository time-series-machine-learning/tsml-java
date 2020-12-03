package tsml.classifiers.distance_based.distances.ed;

import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.data_containers.TimeSeriesInstance;
import weka.core.Instance;

public class EDistance extends BaseDistanceMeasure {
    
    public double distance(final TimeSeriesInstance a, TimeSeriesInstance b, final double limit) {
        double sum = 0;

        int aLength = a.getMaxLength() - 1;
        int bLength = b.getMaxLength() - 1;

        for(int i = 0; i < aLength; i++) {
            sum += DTWDistance.cost(a, i, b, i);
            if(sum > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }

        return sum;
    }
}
