package tsml.data_containers;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Data structure able to handle unequal length, unequally spaced, univariate or
 * multivariate time series.
 */
public class TimeSeriesInstances implements Iterable<TimeSeriesInstance> {

    /* Meta Information */
    boolean isEquallySpaced;
    boolean hasMissing;
    boolean isEqualLength;

    // this could be by dimension, so could be a list.
    int minLength;
    int maxLength;

    public boolean hasMissing() {
        return hasMissing;
    }

    public boolean isEuqallySpaced() {
        return isEquallySpaced;
    }

    public boolean isEqualLength() {
        return isEqualLength;
    }

    public int getMinLength() {
        return minLength;
    }

    public int getMaxLength() {
        return maxLength;
    }

    /* End Meta Information */

    List<TimeSeriesInstance> series_collection;

    // mapping for class labels. so ["apple","orange"] => [0,1]
    // this could be optional for example regression problems.
    String[] classLabels;

    public TimeSeriesInstances() {
        series_collection = new ArrayList<>();
    }

    public TimeSeriesInstances(final String[] classLabels) {
        this();
        setClassLabels(classLabels);
    }

    public TimeSeriesInstances(final List<List<List<Double>>> raw_data) {
        this();

        for (final List<List<Double>> series : raw_data) {
            series_collection.add(new TimeSeriesInstance(series));
        }
    }

    public TimeSeriesInstances(final List<List<List<Double>>> raw_data, final List<Double> label_indexes) {
        this();

        int index = 0;
        for (final List<List<Double>> series : raw_data) {
            series_collection.add(new TimeSeriesInstance(series, label_indexes.get(index++)));
        }

        calculateLengthBounds();
        calculateIfMissing();
    }

    private void calculateLengthBounds() {
        minLength = series_collection.stream().mapToInt(e -> e.minLength).min().getAsInt();
        maxLength = series_collection.stream().mapToInt(e -> e.maxLength).max().getAsInt();
        isEqualLength = minLength == maxLength;
    }

    private void calculateIfMissing() {
        // if any of the instance have a missing value then this is true.
        hasMissing = series_collection.stream().map(e -> e.hasMissing).anyMatch(Boolean::booleanValue);
    }

    public void setClassLabels(final String[] labels) {
        classLabels = labels;
    }

    public String[] getClassLabels() {
        return classLabels;
    }

    public void add(final TimeSeriesInstance new_series) {
        series_collection.add(new_series);

        minLength = Math.min(new_series.minLength, minLength);
        maxLength = Math.min(new_series.maxLength, maxLength);
        hasMissing |= new_series.hasMissing;
        isEqualLength = minLength == maxLength;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder();


        sb.append("Labels: [").append(classLabels[0]);
        for (int i = 1; i < classLabels.length; i++) {
            sb.append(',');
            sb.append(classLabels[i]);
        }
        sb.append(']').append(System.lineSeparator());

        for (final TimeSeriesInstance series : series_collection) {
            sb.append(series.toString());
            sb.append(System.lineSeparator());
        }

        return sb.toString();
    }

    @Override
    public Iterator<TimeSeriesInstance> iterator() {
        return series_collection.iterator();
    }

    public double[][][] toValueArray() {
        final double[][][] output = new double[this.series_collection.size()][][];
        for (int i = 0; i < output.length; ++i) {
            // clone the data so the underlying representation can't be modified
            output[i] = series_collection.get(i).toValueArray();
        }
        return output;
    }

    public TimeSeriesInstance get(final int i) {
        return this.series_collection.get(i);
	}
}
