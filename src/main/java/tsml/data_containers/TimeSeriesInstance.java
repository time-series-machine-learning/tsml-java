package tsml.data_containers;

import org.apache.commons.collections4.list.UnmodifiableList;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Data structure able to store a time series instance. it can be standard
 * (univariate, no missing, equally sampled series) or complex (multivariate,
 * unequal length, unequally spaced, univariate or multivariate time series).
 *
 * Should Instances be immutable after creation? Meta data is calculated on
 * creation, mutability can break this
 */

public class TimeSeriesInstance extends AbstractList<TimeSeries> {

    // reuse the empty class labels in regressed instances
    private final static List<String> EMPTY_LIST = Collections.emptyList();
    
    /* Meta Information */
    private boolean isMultivariate;
    private boolean isEquallySpaced; // todo compute whether timestamps are equally spaced
    private boolean hasMissing;
    private boolean isEqualLength;

    private int minLength;
    private int maxLength;

    public boolean isMultivariate() {
        return isMultivariate;
    }

    public boolean isEquallySpaced() {
        return isEquallySpaced;
    }

    public boolean hasMissing() {
        return hasMissing;
    }

    /** 
     * @return boolean
     */
    public boolean isEqualLength(){
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

    /* End Meta Information */


    /* Data */
    private final List<TimeSeries> seriesDimensions;
    private final int labelIndex;
    private final double targetValue;
    private final List<String> classLabels;
    
    private static ArrayList<TimeSeries> dataToTimeSeries(double[][] data) {
        ArrayList<TimeSeries> series = new ArrayList<>(data.length);
        for(double[] row : data) {
            series.add(new TimeSeries(row));
        }
        return series;
    }
    
    private static ArrayList<TimeSeries> dataToTimeSeries(List<? extends List<Double>> data) {
        ArrayList<TimeSeries> series = new ArrayList<>(data.size());
        for(List<Double> row : data) {
            series.add(new TimeSeries(row));
        }
        return series;
    }
    
    private TimeSeriesInstance(List<TimeSeries> series, TimeSeriesInstance other) {
        // BEWARE! No copy performed, so modification to the given list will change the TSInst. Keep this private, it's only for internal use!
        this.seriesDimensions = series;
        this.targetValue = other.targetValue;
        this.labelIndex = other.labelIndex;
        this.classLabels = other.classLabels;
        
        dataChecks();
    }
    
    public TimeSeriesInstance(List<TimeSeries> series, double targetValue) {
        if(series instanceof UnmodifiableList) {
            this.seriesDimensions = series;
        } else {
            this.seriesDimensions = Collections.unmodifiableList(new ArrayList<>(series));
        }
        this.targetValue = targetValue;
        this.labelIndex = -1;
        this.classLabels = EMPTY_LIST;
        
        dataChecks();
    }
    
    public TimeSeriesInstance(TimeSeriesInstance other) {
        List<TimeSeries> series = other.seriesDimensions;
        if(series instanceof UnmodifiableList) {
            this.seriesDimensions = series;
        } else {
            this.seriesDimensions = Collections.unmodifiableList(new ArrayList<>(series));
        }
        List<String> classLabels = other.classLabels;
        if(other.classLabels instanceof UnmodifiableList) {
            this.classLabels = classLabels;
        } else {
            this.classLabels = Collections.unmodifiableList(classLabels);
        }
        this.targetValue = other.targetValue;
        this.labelIndex = other.labelIndex;
        
        dataChecks();
    }

    public TimeSeriesInstance(List<TimeSeries> series, int labelIndex, List<String> classLabels) {
        if(series instanceof UnmodifiableList) {
            this.seriesDimensions = series;
        } else {
            this.seriesDimensions = Collections.unmodifiableList(new ArrayList<>(series));
        }
        if(classLabels instanceof UnmodifiableList) {
            this.classLabels = classLabels;
        } else {
            this.classLabels = Collections.unmodifiableList(classLabels);
        }
        this.targetValue = labelIndex;
        this.labelIndex = labelIndex;
        
        dataChecks();
    }
    
    public TimeSeriesInstance(List<TimeSeries> series, String label, List<String> classLabels) {
        this(series, classLabels.indexOf(label), classLabels);
    }

    public TimeSeriesInstance(List<TimeSeries> series) {
        if(series instanceof UnmodifiableList) {
            this.seriesDimensions = series;
        } else {
            this.seriesDimensions = Collections.unmodifiableList(new ArrayList<>(series));
        }
        this.targetValue = 0;
        this.labelIndex = -1;
        this.classLabels = EMPTY_LIST;
        
        dataChecks();
    }
    
    public static TimeSeriesInstance fromLabelledData(double[][] data, int labelIndex, List<String> labels) {
        return new TimeSeriesInstance(dataToTimeSeries(data), labelIndex, labels);
    }

    public static TimeSeriesInstance fromLabelledData(double[][] data, String label, List<String> labels) {
        return new TimeSeriesInstance(dataToTimeSeries(data), label, labels);
    }
    
    public static TimeSeriesInstance fromRegressedData(double[][] data, double targetValue) {
        return new TimeSeriesInstance(dataToTimeSeries(data), targetValue);
    }
    
    public static TimeSeriesInstance fromData(double[][] data) {
        return new TimeSeriesInstance(dataToTimeSeries(data));
    }

    public static TimeSeriesInstance fromLabelledData(List<? extends List<Double>> data, int labelIndex, List<String> labels) {
        return new TimeSeriesInstance(dataToTimeSeries(data), labelIndex, labels);
    }

    public static TimeSeriesInstance fromLabelledData(List<? extends List<Double>> data, String label, List<String> labels) {
        return new TimeSeriesInstance(dataToTimeSeries(data), label, labels);
    }

    public static TimeSeriesInstance fromRegressedData(List<? extends List<Double>> data, double targetValue) {
        return new TimeSeriesInstance(dataToTimeSeries(data), targetValue);
    }

    public static TimeSeriesInstance fromData(List<? extends List<Double>> data) {
        return new TimeSeriesInstance(dataToTimeSeries(data));
    }
    public List<String> getClassLabels() {
        return classLabels;
    }
    
    private void dataChecks(){

        // check info is in the expected format
        Objects.requireNonNull(seriesDimensions);
        Objects.requireNonNull(classLabels);
        if(labelIndex > classLabels.size() - 1) {
            throw new IndexOutOfBoundsException("label index " + labelIndex + " is out of range of " + classLabels);
        }
        if(classLabels.isEmpty()) {
            if(labelIndex != -1) {
                throw new IllegalStateException("label index set to " + labelIndex + " but labels is empty");
            }
        } else {
            if(labelIndex < 0) {
                throw new IndexOutOfBoundsException("label index " + labelIndex + " is less than zero");
            }
        }
        // check immutability
        if(!(classLabels instanceof UnmodifiableList)) {
            throw new IllegalStateException("class labels are mutable");
        }
        if(!(seriesDimensions instanceof UnmodifiableList)) {
            throw new IllegalStateException("dimensions are mutable");
        }

        calculateIfMultivariate();
        calculateLengthBounds();
        calculateIfMissing();
    }
    
    private void calculateIfMultivariate(){
        isMultivariate = seriesDimensions.size() > 1;
    }

	private void calculateLengthBounds() {
        minLength = seriesDimensions.stream().mapToInt(TimeSeries::getSeriesLength).min().getAsInt();
        maxLength = seriesDimensions.stream().mapToInt(TimeSeries::getSeriesLength).max().getAsInt();
        isEqualLength = minLength == maxLength;
    }

    private void calculateIfMissing() {
        // if any of the series have a NaN value, across all dimensions then this is
        // true.
        hasMissing = seriesDimensions.stream().anyMatch(e -> e.streamValues().anyMatch(Double::isNaN));
    };

    
    /** 
     * @return int
     */
    public int getNumDimensions() {
        return seriesDimensions.size();
    }

    
    /** 
     * @return int
     */
    public int getLabelIndex(){
        return labelIndex;
    }
    
    /** 
     * @param index
     * @return List<Double>
     */
    public List<Double> getVSliceList(int index){
        List<Double> out = new ArrayList<>(getNumDimensions());
        for(TimeSeries ts : seriesDimensions){
            out.add(ts.getValue(index));
        }

        return out;
    }
    
    /** 
     * @param index
     * @return double[]
     */
    public double[] getVSliceArray(int index){
        double[] out = new double[getNumDimensions()];
        int i=0;
        for(TimeSeries ts : seriesDimensions){
            out[i++] = ts.getValue(index);
        }

        return out;
    }
    
    /** 
     * @param indexesToKeep
     * @return List<List<Double>>
     */
    public List<List<Double>> getVSliceList(int[] indexesToKeep){
        return getVSliceList(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    
    /** 
     * @param indexesToKeep
     * @return List<List<Double>>
     */
    public List<List<Double>> getVSliceList(List<Integer> indexesToKeep){
        List<List<Double>> out = new ArrayList<>(getNumDimensions());
        for(TimeSeries ts : seriesDimensions){
            out.add(ts.getVSliceList(indexesToKeep));
        }

        return out;
    }

 
    
    /** 
     * @param indexesToKeep
     * @return double[][]
     */
    public double[][] getVSliceArray(int[] indexesToKeep){
        return getVSliceArray(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    
    /** 
     * @param indexesToKeep
     * @return double[][]
     */
    public double[][] getVSliceArray(List<Integer> indexesToKeep){
        double[][] out = new double[getNumDimensions()][];
        int i=0;
        for(TimeSeries ts : seriesDimensions){
            out[i++] = ts.getVSliceArray(indexesToKeep);
        }

        return out;
    }
    
    public TimeSeriesInstance getVSlice(List<Integer> indexesToKeep) {
        return new TimeSeriesInstance(dataToTimeSeries(getVSliceArray(indexesToKeep)), this);
    }
    
    public TimeSeriesInstance getVSlice(int[] indexesToKeep) {
        return getVSlice(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }
    
    public TimeSeriesInstance getVSlice(int index) {
        return getVSlice(new int[] {index});
    }


    
    /** 
     * @param dim
     * @return List<Double>
     */
    public List<Double> getHSliceList(int dim){
        return seriesDimensions.get(dim).getSeries();
    }

    
    /** 
     * @param dim
     * @return double[]
     */
    public double[] getHSliceArray(int dim){
        return seriesDimensions.get(dim).toValueArray();
    }

    
    /** 
     * @param dimensionsToKeep
     * @return List<List<Double>>
     */
    public List<List<Double>> getHSliceList(int[] dimensionsToKeep){
        return getHSliceList(Arrays.stream(dimensionsToKeep).boxed().collect(Collectors.toList()));
    }

    
    /** 
     * TODO: not a clone. may need to be careful...
     * @param dimensionsToKeep
     * @return List<List<Double>>
     */
    public List<List<Double>> getHSliceList(List<Integer> dimensionsToKeep){
        List<List<Double>> out = new ArrayList<>(dimensionsToKeep.size());
        for(Integer dim : dimensionsToKeep)
            out.add(seriesDimensions.get(dim).getSeries());

        return out;
    }

    
    /** 
     * @param dimensionsToKeep
     * @return double[][]
     */
    public double[][] getHSliceArray(int[] dimensionsToKeep){
        return getHSliceArray(Arrays.stream(dimensionsToKeep).boxed().collect(Collectors.toList()));
    }

    
    /** 
     * @param dimensionsToKeep
     * @return double[][]
     */
    public double[][] getHSliceArray(List<Integer> dimensionsToKeep){
        double[][] out = new double[dimensionsToKeep.size()][];
        int i=0;
        for(Integer dim : dimensionsToKeep){
            out[i++] = seriesDimensions.get(dim).toValueArray();
        }

        return out;
    }
    
    public TimeSeriesInstance getHSlice(List<Integer> dimensionsToKeep) {
        return new TimeSeriesInstance(dataToTimeSeries(getHSliceArray(dimensionsToKeep)), this);
    }
    
    public TimeSeriesInstance getHSlice(int[] dimensionsToKeep) {
        return getHSlice(Arrays.stream(dimensionsToKeep).boxed().collect(Collectors.toList()));
    }
    
    public TimeSeriesInstance getHSlice(int index) {
        return getHSlice(new int[] {index});
    }

    
    /** 
     * @return String
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append("Num Dimensions: ").append(getNumDimensions()).append(" Class Label Index: ").append(labelIndex);
        for (TimeSeries ts : seriesDimensions) {
            sb.append(System.lineSeparator());
            sb.append(ts.toString());
        }

        return sb.toString();
    }
    
    /** 
     * @return double[][]
     */
    public double[][] toValueArray(){
        double[][] output = new double[this.seriesDimensions.size()][];
        for (int i=0; i<output.length; ++i){
             //clone the data so the underlying representation can't be modified
            output[i] = seriesDimensions.get(i).toValueArray();
        }
        return output;
    }

    
    /** 
     * @return double[][]
     */
    public double[][] toTransposedArray(){
        return this.getVSliceArray(IntStream.range(0, maxLength).toArray());
    }
	
    /** 
     * @param i
     * @return TimeSeries
     */
    public TimeSeries get(int i) {
        return this.seriesDimensions.get(i);
	}
	

    public double getTargetValue() {
        return targetValue;
    }

    @Override public int size() {
        return getNumDimensions();
    }

    @Override public void add(final int i, final TimeSeries doubles) {
        throw new UnsupportedOperationException("TimeSeriesInstance not mutable");
    }

    @Override public TimeSeries set(final int i, final TimeSeries doubles) {
        throw new UnsupportedOperationException("TimeSeriesInstance not mutable");
    }

    @Override public void clear() {
        throw new UnsupportedOperationException("TimeSeriesInstance not mutable");
    }

    @Override public TimeSeries remove(final int i) {
        throw new UnsupportedOperationException("TimeSeriesInstance not mutable");
    }
}
