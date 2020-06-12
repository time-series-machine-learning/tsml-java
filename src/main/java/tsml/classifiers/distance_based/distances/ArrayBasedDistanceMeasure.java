package tsml.classifiers.distance_based.distances;

import tsml.transformers.Indexer;
import weka.core.Instance;

public abstract class ArrayBasedDistanceMeasure extends BaseDistanceMeasure {

    protected final double findDistance(final Instance ai, final Instance bi, final double limit) {
        final double[] a = Indexer.extractAttributeValuesAndClassLabel(ai);
        final double[] b = Indexer.extractAttributeValuesAndClassLabel(bi);
        return distance(a, b, limit);
    }

    public final double distance(final double[] a, final double[] b, final double limit) {
        return findDistance(a, b, limit);
    }

    protected abstract double findDistance(final double[] a, final double[] b, final double limit);
}
