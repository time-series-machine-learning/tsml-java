package tsml.classifiers.distance_based.distances;

import tsml.classifiers.distance_based.utils.instance.ExposedDenseInstance;
import weka.core.Instance;

public abstract class ArrayBasedDistanceMeasure extends BaseDistanceMeasure {

    public double distance(final Instance ai, final Instance bi, final double limit) {
        checkData(ai, bi);
        final double[] a = ExposedDenseInstance.extractAttributeValuesAndClassLabel(ai);
        final double[] b = ExposedDenseInstance.extractAttributeValuesAndClassLabel(bi);
        return distance(a, b, limit);
    }

    public abstract double distance(double[] a, double[] b, final double limit);
}
