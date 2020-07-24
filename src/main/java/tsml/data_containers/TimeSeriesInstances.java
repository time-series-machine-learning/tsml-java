package tsml.data_containers;

import java.util.ArrayList;
import java.util.Arrays;
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

    int[] classCounts;

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
            //using the add function means all stats should be correctly counted.
            series_collection.add(new TimeSeriesInstance(series, label_indexes.get(index++)));
        }

        calculateLengthBounds();
        calculateIfMissing();
    }

    public TimeSeriesInstances(final double[][][] raw_data) {
        this();

        for (final double[][] series : raw_data) {
            //using the add function means all stats should be correctly counted.
            series_collection.add(new TimeSeriesInstance(series));
        }

        calculateLengthBounds();
        calculateIfMissing();
    }

    public TimeSeriesInstances(final double[][][] raw_data, int[] label_indexes) {
        this();

        int index = 0;
        for (double[][] series : raw_data) {
            //using the add function means all stats should be correctly counted.
            series_collection.add(new TimeSeriesInstance(series, label_indexes[index++]));
        }

        calculateLengthBounds();
        calculateIfMissing();


    }


    private void calculateClassCounts() {
        classCounts = new int[classLabels.length];
        for(TimeSeriesInstance inst : series_collection){
            classCounts[inst.classLabelIndex]++;
        }
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

    public void setClassLabels(String[] labels) {
        classLabels = labels;

        calculateClassCounts();
    }

    public String[] getClassLabels() {
        return classLabels;
    }

    public int[] getClassCounts(){
        return classCounts;
    }

    public void add(final TimeSeriesInstance new_series) {
        series_collection.add(new_series);

        //guard for if we're going to force update classCounts after.
        if(classCounts != null && new_series.classLabelIndex < classCounts.length)
            classCounts[new_series.classLabelIndex]++;

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
        final double[][][] output = new double[series_collection.size()][][];
        for (int i = 0; i < output.length; ++i) {
            // clone the data so the underlying representation can't be modified
            output[i] = series_collection.get(i).toValueArray();
        }
        return output;
    }

    public int[] getClassIndexes(){
        int[] out = new int[numInstances()];
        int index=0;
        for(TimeSeriesInstance inst : series_collection){
            out[index++] = inst.classLabelIndex;
        }
        return out;
    }

    //assumes equal numbers of channels
    public double[] getSingleSliceArray(int index){
        double[] out = new double[numInstances() * series_collection.get(0).getNumChannels()];
        int i=0;
        for(TimeSeriesInstance inst : series_collection){
            for(TimeSeries ts : inst)
                // if the index isn't always valid, populate with NaN values.
                out[i++] = ts.hasValidValueAt(index) ? ts.get(index) : Double.NaN;
        }

        return out;
    }

    public List<List<List<Double>>> getSliceList(int[] indexesToKeep){
        return getSliceList(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    public List<List<List<Double>>> getSliceList(List<Integer> indexesToKeep){
        List<List<List<Double>>> out = new ArrayList<>(numInstances());
        for(TimeSeriesInstance inst : series_collection){
            out.add(inst.getSliceList(indexesToKeep));
        }

        return out;
    }

    public double[][][] getSliceArray(int[] indexesToKeep){
        return getSliceArray(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    public double[][][] getSliceArray(List<Integer> indexesToKeep){
        double[][][] out = new double[numInstances()][][];
        int i=0;
        for(TimeSeriesInstance inst : series_collection){
            out[i++] = inst.getSliceArray(indexesToKeep);
        }

        return out;
    }

    public TimeSeriesInstance get(final int i) {
        return series_collection.get(i);
	}

	public int numInstances() {
		return series_collection.size();
    }
    
}
