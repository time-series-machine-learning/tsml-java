package tsml.classifiers.distance_based.distances;

import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;

import java.util.Enumeration;

public class DistanceFunctionAdapter implements DistanceFunction {

    public DistanceFunctionAdapter(final DistanceMeasure dm) {
        this.dm = dm;
    }

    private final DistanceMeasure dm;

    public boolean isSymmetric() {
        return dm.isSymmetric();
    }

    @Override public double distance(final Instance a, final Instance b) {
        return dm.distance(a, b);
    }

    @Override public double distance(final Instance a, final Instance b,
            final PerformanceStats stats) {
        return dm.distance(a, b);
    }

    @Override public double distance(final Instance a, final Instance b, final double limit,
            final PerformanceStats stats) {
        return dm.distance(a, b, limit);
    }

    @Override public void postProcessDistances(final double[] distances) {

    }

    @Override public void update(final Instance ins) {

    }

    @Override public double distance(final Instance a, final Instance b, final double limit) {
        return dm.distance(a, b, limit);
    }

    @Override public String toString() {
        return dm.toString();
    }

    public DistanceFunction asDistanceFunction() {
        return this;
    }

    @Override public String[] getOptions() {
        return dm.getOptions();
    }

    @Override public Enumeration listOptions() {
        return dm.listOptions();
    }

    @Override public void setOptions(final String[] options) throws Exception {
        dm.setOptions(options);
    }

    @Override public void setInstances(final Instances insts) {
        dm.buildDistanceMeasure(insts);
    }

    @Override public Instances getInstances() {
        return null;
    }

    @Override public void setAttributeIndices(final String value) {

    }

    @Override public String getAttributeIndices() {
        return null;
    }

    @Override public void setInvertSelection(final boolean value) {

    }

    @Override public boolean getInvertSelection() {
        return false;
    }
}
