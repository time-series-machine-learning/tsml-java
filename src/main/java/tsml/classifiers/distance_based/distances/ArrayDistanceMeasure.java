//package tsml.classifiers.distance_based.distances;
//
//import tsml.classifiers.distance_based.utils.instance.ExposedDenseInstance;
//import weka.core.Instance;
//
//public abstract class ArrayDistanceMeasure extends BaseDistanceMeasure {
//
//    @Override
//    protected double findDistance(final Instance ai, final Instance bi, final double limit) {
//        final double[] a = ExposedDenseInstance.extractAttributeValuesAndClassLabel(ai);
//        final double[] b = ExposedDenseInstance.extractAttributeValuesAndClassLabel(bi);
//        return findDistance(a, b, limit);
//    }
//
//    protected abstract double findDistance(double[] a, double[] b, final double limit);
//}
