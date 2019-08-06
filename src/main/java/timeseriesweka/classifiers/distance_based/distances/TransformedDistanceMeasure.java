package timeseriesweka.classifiers.distance_based.distances;

import timeseriesweka.classifiers.distance_based.distances.DistanceMeasure;
import timeseriesweka.filters.Cache;

public class TransformedDistanceMeasure
    extends DistanceMeasure {


    private final Cache<double[], double[]> cache;
    private final DistanceMeasure distanceMeasure;

    @Override
    public double distance() {
        double[] originalA = getTarget();
        double[] originalB = getCandidate();
        double[] a = cache.get(originalA);
        double[] b = cache.get(originalB);
        setTarget(a);
        setCandidate(b);
        double distance = distanceMeasure.distance();
        setTarget(originalA);
        setCandidate(originalB);
        return distance;
    }

    @Override
    public String toString() {
        return distanceMeasure.toString();
    }

    public TransformedDistanceMeasure(final Cache<double[], double[]> cache,
                                      final DistanceMeasure distanceMeasure
                                     ) {
        this.cache = cache;
        this.distanceMeasure = distanceMeasure;
    }

    @Override
    public void setOption(final String key, final String value) {
        distanceMeasure.setOption(key, value);
    }

    @Override
    public void setOptions(final String[] options) throws
                                                   Exception {
        distanceMeasure.setOptions(options);
    }

    @Override
    public String[] getOptions() {
        return distanceMeasure.getOptions();
    }
}
