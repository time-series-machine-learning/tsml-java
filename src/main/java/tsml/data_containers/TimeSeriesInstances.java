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
    String description;
    String problemName;
    boolean isEquallySpaced = true;
    boolean hasMissing;
    boolean isEqualLength;

    boolean isMultivariate;
    boolean hasTimeStamps;

    // this could be by dimension, so could be a list.
    int minLength;
    int maxLength;
    int maxNumChannels;


	public String getProblemName() {
		return problemName;
	}

	public boolean hasTimeStamps() {
		return hasTimeStamps;
	}

    public boolean hasMissing() {
        return hasMissing;
    }

    public boolean isEquallySpaced() {
        return isEquallySpaced;
    }

    public boolean isMultivariate(){
        return isMultivariate;
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

    public int numClasses(){
        return classLabels.length;
    }

	public int getMaxNumChannels() {
		return maxNumChannels;
	}

    public void setProblemName(String problemName) {
        this.problemName = problemName;
    }

    public void setEquallySpaced(boolean isEquallySpaced) {
        this.isEquallySpaced = isEquallySpaced;
    }

    public boolean isHasTimeStamps() {
        return hasTimeStamps;
    }

    public void setHasTimeStamps(boolean hasTimeStamps) {
        this.hasTimeStamps = hasTimeStamps;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
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

        dataChecks();
    }

    public TimeSeriesInstances(final double[][][] raw_data) {
        this();

        for (final double[][] series : raw_data) {
            //using the add function means all stats should be correctly counted.
            series_collection.add(new TimeSeriesInstance(series));
        }

        dataChecks();
    }

    public TimeSeriesInstances(final double[][][] raw_data, int[] label_indexes) {
        this();

        int index = 0;
        for (double[][] series : raw_data) {
            //using the add function means all stats should be correctly counted.
            series_collection.add(new TimeSeriesInstance(series, label_indexes[index++]));
        }

        dataChecks();
    }

    public TimeSeriesInstances(final double[][][] raw_data, int[] label_indexes, String[] labels) {
        this(raw_data, label_indexes);
        classLabels = labels;
    }

    public TimeSeriesInstances(List<TimeSeriesInstance> data, String[] labels) {
        this();
        
        for(TimeSeriesInstance d : data)
            series_collection.add(d);

        classLabels = labels;

        dataChecks();
	}

	private void dataChecks(){
        calculateLengthBounds();
        calculateIfMissing();
        calculateIfMultivariate();
        calculateNumChannels();
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

    private void calculateNumChannels(){
        maxNumChannels = series_collection.stream().mapToInt(e -> e.getNumChannels()).max().getAsInt();
    }
    
    private void calculateIfMultivariate(){
        isMultivariate = series_collection.stream().map(e -> e.isMultivariate).anyMatch(Boolean::booleanValue);
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

    public String getClassLabelsFormatted(){
        String output = " ";
        for(String s : classLabels)
            output += s + " ";
        return output;
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
    public double[] getVSliceArray(int index){
        double[] out = new double[numInstances() * series_collection.get(0).getNumChannels()];
        int i=0;
        for(TimeSeriesInstance inst : series_collection){
            for(TimeSeries ts : inst)
                // if the index isn't always valid, populate with NaN values.
                out[i++] = ts.hasValidValueAt(index) ? ts.get(index) : Double.NaN;
        }

        return out;
    }

    public List<List<List<Double>>> getVSliceList(int[] indexesToKeep){
        return getVSliceList(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    public List<List<List<Double>>> getVSliceList(List<Integer> indexesToKeep){
        List<List<List<Double>>> out = new ArrayList<>(numInstances());
        for(TimeSeriesInstance inst : series_collection){
            out.add(inst.getVSliceList(indexesToKeep));
        }

        return out;
    }

    public double[][][] getVSliceArray(int[] indexesToKeep){
        return getVSliceArray(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    public double[][][] getVSliceArray(List<Integer> indexesToKeep){
        double[][][] out = new double[numInstances()][][];
        int i=0;
        for(TimeSeriesInstance inst : series_collection){
            out[i++] = inst.getVSliceArray(indexesToKeep);
        }

        return out;
    }

    //assumes equal numbers of channels
    public double[][] getHSliceArray(int dim){
        double[][] out = new double[numInstances()][];
        int i=0;
        for(TimeSeriesInstance inst : series_collection){
            // if the index isn't always valid, populate with NaN values.
            out[i++] = inst.getSingleHSliceArray(dim);
        }
        return out;
    }
    
    public List<List<List<Double>>> getHSliceList(int[] indexesToKeep){
        return getVSliceList(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }
    
    public List<List<List<Double>>> getHSliceList(List<Integer> indexesToKeep){
        List<List<List<Double>>> out = new ArrayList<>(numInstances());
        for(TimeSeriesInstance inst : series_collection){
            out.add(inst.getHSliceList(indexesToKeep));
        }

        return out;
    }
    
    public double[][][] getHSliceArray(int[] indexesToKeep){
        return getHSliceArray(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    public double[][][] getHSliceArray(List<Integer> indexesToKeep){
        double[][][] out = new double[numInstances()][][];
        int i=0;
        for(TimeSeriesInstance inst : series_collection){
            out[i++] = inst.getHSliceArray(indexesToKeep);
        }

        return out;
    }

    public TimeSeriesInstance get(final int i) {
        return series_collection.get(i);
    }
    
    public List<TimeSeriesInstance> getAll(){
        return series_collection;
    }

	public int numInstances() {
		return series_collection.size();
    }




    
}
