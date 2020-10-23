package tsml.data_containers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
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
    int maxNumDimensions;


	
    /** 
     * @return String
     */
    public String getProblemName() {
		return problemName;
	}

	
    /** 
     * @return boolean
     */
    public boolean hasTimeStamps() {
		return hasTimeStamps;
	}

    
    /** 
     * @return boolean
     */
    public boolean hasMissing() {
        return hasMissing;
    }

    
    /** 
     * @return boolean
     */
    public boolean isEquallySpaced() {
        return isEquallySpaced;
    }

    
    /** 
     * @return boolean
     */
    public boolean isMultivariate(){
        return isMultivariate;
    }

    
    /** 
     * @return boolean
     */
    public boolean isEqualLength() {
        return isEqualLength;
    }

    
    /** 
     * @return int
     */
    public int getMinLength() {
        return minLength;
    }

    
    /** 
     * @return int
     */
    public int getMaxLength() {
        return maxLength;
    }

    
    /** 
     * @return int
     */
    public int numClasses(){
        return classLabels.length;
    }

	
    /** 
     * @return int
     */
    public int getMaxNumChannels() {
		return maxNumDimensions;
	}

    
    /** 
     * @param problemName
     */
    public void setProblemName(String problemName) {
        this.problemName = problemName;
    }

    
    /** 
     * @param isEquallySpaced
     */
    public void setEquallySpaced(boolean isEquallySpaced) {
        this.isEquallySpaced = isEquallySpaced;
    }

    
    /** 
     * @return boolean
     */
    public boolean isHasTimeStamps() {
        return hasTimeStamps;
    }

    
    /** 
     * @param hasTimeStamps
     */
    public void setHasTimeStamps(boolean hasTimeStamps) {
        this.hasTimeStamps = hasTimeStamps;
    }

    
    /** 
     * @return String
     */
    public String getDescription() {
        return description;
    }

    
    /** 
     * @param description
     */
    public void setDescription(String description) {
        this.description = description;
    }

    /* End Meta Information */

    List<TimeSeriesInstance> seriesCollection;

    // mapping for class labels. so ["apple","orange"] => [0,1]
    // this could be optional for example regression problems.
    String[] classLabels;

    int[] classCounts;

    public TimeSeriesInstances() {
        seriesCollection = new ArrayList<>();
    }

    public TimeSeriesInstances(final String[] classLabels) {
        this();
        setClassLabels(classLabels);
    }

    public TimeSeriesInstances(final List<List<List<Double>>> raw_data) {
        this();

        for (final List<List<Double>> series : raw_data) {
            seriesCollection.add(new TimeSeriesInstance(series));
        }

        dataChecks();
    }

    
    public TimeSeriesInstances(final List<List<List<Double>>> raw_data, final List<Double> label_indexes) {
        this();

        int index = 0;
        for (final List<List<Double>> series : raw_data) {
            //using the add function means all stats should be correctly counted.
            seriesCollection.add(new TimeSeriesInstance(series, label_indexes.get(index++)));
        }

        dataChecks();
    }

    public TimeSeriesInstances(final double[][][] raw_data) {
        this();

        for (final double[][] series : raw_data) {
            //using the add function means all stats should be correctly counted.
            seriesCollection.add(new TimeSeriesInstance(series));
        }

        dataChecks();
    }

    public TimeSeriesInstances(final double[][][] raw_data, int[] label_indexes) {
        this();

        int index = 0;
        for (double[][] series : raw_data) {
            //using the add function means all stats should be correctly counted.
            seriesCollection.add(new TimeSeriesInstance(series, label_indexes[index++]));
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
            seriesCollection.add(d);

        classLabels = labels;

        dataChecks();
	}

	private void dataChecks(){
        calculateLengthBounds();
        calculateIfMissing();
        calculateIfMultivariate();
        calculateNumDimensions();
    }

    private void calculateClassCounts() {
        classCounts = new int[classLabels.length];
        for(TimeSeriesInstance inst : seriesCollection){
            classCounts[inst.classLabelIndex]++;
        }
    }

    private void calculateLengthBounds() {
        minLength = seriesCollection.stream().mapToInt(e -> e.minLength).min().getAsInt();
        maxLength = seriesCollection.stream().mapToInt(e -> e.maxLength).max().getAsInt();
        isEqualLength = minLength == maxLength;
    }

    private void calculateNumDimensions(){
        maxNumDimensions = seriesCollection.stream().mapToInt(e -> e.getNumDimensions()).max().getAsInt();
    }
    
    private void calculateIfMultivariate(){
        isMultivariate = seriesCollection.stream().map(e -> e.isMultivariate).anyMatch(Boolean::booleanValue);
    }

    private void calculateIfMissing() {
        // if any of the instance have a missing value then this is true.
        hasMissing = seriesCollection.stream().map(e -> e.hasMissing).anyMatch(Boolean::booleanValue);
    }

    
    /** 
     * @param labels
     */
    public void setClassLabels(String[] labels) {
        classLabels = labels;

        calculateClassCounts();
    }

    
    /** 
     * @return String[]
     */
    public String[] getClassLabels() {
        return classLabels;
    }

    
    /** 
     * @return String
     */
    public String getClassLabelsFormatted(){
        String output = " ";
        for(String s : classLabels)
            output += s + " ";
        return output;
    }

    
    /** 
     * @return int[]
     */
    public int[] getClassCounts(){
        return classCounts;
    }

    
    /** 
     * @param new_series
     */
    public void add(final TimeSeriesInstance new_series) {
        seriesCollection.add(new_series);

        //guard for if we're going to force update classCounts after.
        if(classCounts != null && new_series.classLabelIndex < classCounts.length)
            classCounts[new_series.classLabelIndex]++;

        minLength = Math.min(new_series.minLength, minLength);
        maxLength = Math.max(new_series.maxLength, maxLength);
        maxNumDimensions = Math.max(new_series.getNumDimensions(), maxNumDimensions);
        hasMissing |= new_series.hasMissing;
        isEqualLength = minLength == maxLength;
        isMultivariate |= new_series.isMultivariate;
    }

    
    /** 
     * @return String
     */
    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder();


        sb.append("Labels: [").append(classLabels[0]);
        for (int i = 1; i < classLabels.length; i++) {
            sb.append(',');
            sb.append(classLabels[i]);
        }
        sb.append(']').append(System.lineSeparator());

        for (final TimeSeriesInstance series : seriesCollection) {
            sb.append(series.toString());
            sb.append(System.lineSeparator());
        }

        return sb.toString();
    }

    
    /** 
     * @return Iterator<TimeSeriesInstance>
     */
    @Override
    public Iterator<TimeSeriesInstance> iterator() {
        return seriesCollection.iterator();
    }

    
    /** 
     * @return double[][][]
     */
    public double[][][] toValueArray() {
        final double[][][] output = new double[seriesCollection.size()][][];
        for (int i = 0; i < output.length; ++i) {
            // clone the data so the underlying representation can't be modified
            output[i] = seriesCollection.get(i).toValueArray();
        }
        return output;
    }

    
    /** 
     * @return int[]
     */
    public int[] getClassIndexes(){
        int[] out = new int[numInstances()];
        int index=0;
        for(TimeSeriesInstance inst : seriesCollection){
            out[index++] = inst.classLabelIndex;
        }
        return out;
    }

    
    /** 
     * @param index
     * @return double[]
     */
    //assumes equal numbers of channels
    public double[] getVSliceArray(int index){
        double[] out = new double[numInstances() * seriesCollection.get(0).getNumDimensions()];
        int i=0;
        for(TimeSeriesInstance inst : seriesCollection){
            for(TimeSeries ts : inst)
                // if the index isn't always valid, populate with NaN values.
                out[i++] = ts.hasValidValueAt(index) ? ts.getValue(index) : Double.NaN;
        }

        return out;
    }

    
    /** 
     * @param indexesToKeep
     * @return List<List<List<Double>>>
     */
    public List<List<List<Double>>> getVSliceList(int[] indexesToKeep){
        return getVSliceList(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    
    /** 
     * @param indexesToKeep
     * @return List<List<List<Double>>>
     */
    public List<List<List<Double>>> getVSliceList(List<Integer> indexesToKeep){
        List<List<List<Double>>> out = new ArrayList<>(numInstances());
        for(TimeSeriesInstance inst : seriesCollection){
            out.add(inst.getVSliceList(indexesToKeep));
        }

        return out;
    }

    
    /** 
     * @param indexesToKeep
     * @return double[][][]
     */
    public double[][][] getVSliceArray(int[] indexesToKeep){
        return getVSliceArray(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    
    /** 
     * @param indexesToKeep
     * @return double[][][]
     */
    public double[][][] getVSliceArray(List<Integer> indexesToKeep){
        double[][][] out = new double[numInstances()][][];
        int i=0;
        for(TimeSeriesInstance inst : seriesCollection){
            out[i++] = inst.getVSliceArray(indexesToKeep);
        }

        return out;
    }

    
    /** 
     * @param dim
     * @return double[][]
     */
    //assumes equal numbers of channels
    public double[][] getHSliceArray(int dim){
        double[][] out = new double[numInstances()][];
        int i=0;
        for(TimeSeriesInstance inst : seriesCollection){
            // if the index isn't always valid, populate with NaN values.
            out[i++] = inst.getSingleHSliceArray(dim);
        }
        return out;
    }
    
    
    /** 
     * @param indexesToKeep
     * @return List<List<List<Double>>>
     */
    public List<List<List<Double>>> getHSliceList(int[] indexesToKeep){
        return getVSliceList(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }
    
    
    /** 
     * @param indexesToKeep
     * @return List<List<List<Double>>>
     */
    public List<List<List<Double>>> getHSliceList(List<Integer> indexesToKeep){
        List<List<List<Double>>> out = new ArrayList<>(numInstances());
        for(TimeSeriesInstance inst : seriesCollection){
            out.add(inst.getHSliceList(indexesToKeep));
        }

        return out;
    }
    
    
    /** 
     * @param indexesToKeep
     * @return double[][][]
     */
    public double[][][] getHSliceArray(int[] indexesToKeep){
        return getHSliceArray(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    
    /** 
     * @param indexesToKeep
     * @return double[][][]
     */
    public double[][][] getHSliceArray(List<Integer> indexesToKeep){
        double[][][] out = new double[numInstances()][][];
        int i=0;
        for(TimeSeriesInstance inst : seriesCollection){
            out[i++] = inst.getHSliceArray(indexesToKeep);
        }

        return out;
    }

    
    /** 
     * @param i
     * @return TimeSeriesInstance
     */
    public TimeSeriesInstance get(final int i) {
        return seriesCollection.get(i);
    }
    
    
    /** 
     * @return List<TimeSeriesInstance>
     */
    public List<TimeSeriesInstance> getAll(){
        return seriesCollection;
    }

	
    /** 
     * @return int
     */
    public int numInstances() {
		return seriesCollection.size();
    }

    
    /** 
     * @return int
     */
    @Override
    public int hashCode(){
        return this.seriesCollection.hashCode();
    }


    public Map<Integer, Integer> getHistogramOfLengths(){
        Map<Integer, Integer> out = new TreeMap<>();
        for(TimeSeriesInstance inst : seriesCollection){
            for(TimeSeries ts : inst){
                out.merge(ts.getSeriesLength(), 1, Integer::sum);
            }
        }

        return out;
    }

    
}
