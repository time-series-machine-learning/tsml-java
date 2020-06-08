package tsml.classifiers.distance_based.distances.ed;

import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.utils.instance.ExposedDenseInstance;
import utilities.Utilities;
import weka.core.Instance;

public class EuclideanDistance extends BaseDistanceMeasure {

    @Override
    public double distance(final Instance ai, final Instance bi, final double limit) {
        checkData(ai, bi);

        double sum = 0;

        double[] a = ExposedDenseInstance.extractAttributeValuesAndClassLabel(ai);
        double[] b = ExposedDenseInstance.extractAttributeValuesAndClassLabel(bi);

        int aLength = a.length - 1;
        int bLength = b.length - 1;

        // put a or first as the longest time series
        if(bLength > aLength) {
            double[] tmp = a;
            a = b;
            b = tmp;
            int tmpLength = aLength;
            aLength = bLength;
            bLength = tmpLength;
        }

        for(int i = 0; i < aLength; i++) {
            sum += Utilities.squaredDifference(a, i, b, i);
        }

        return sum;
    }
}
