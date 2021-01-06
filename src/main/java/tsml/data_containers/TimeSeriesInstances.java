package tsml.data_containers;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import static tsml.data_containers.TimeSeriesInstance.EMPTY_CLASS_LABELS;

/**
 * Data structure able to handle unequal length, unequally spaced, univariate or
 * multivariate time series.
 */
public class TimeSeriesInstances implements Iterable<TimeSeriesInstance> {

    /* Meta Information */
    private String description;
    private String problemName;
    private boolean isEquallySpaced = true;
    private boolean hasMissing;
    private boolean isEqualLength;

    private boolean isMultivariate;
    private boolean hasTimeStamps;

    // this could be by dimension, so could be a list.
    private int minLength;
    private int maxLength;
    private int maxNumDimensions;

    public int getMaxNumDimensions() {
        return maxNumDimensions;
    }

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

    private List<TimeSeriesInstance> seriesCollection = new ArrayList<>();

    // mapping for class labels. so ["apple","orange"] => [0,1]
    // this could be optional for example regression problems.
    private String[] classLabels = EMPTY_CLASS_LABELS;

    private int[] classCounts;

    public TimeSeriesInstances(final String[] classLabels) {        
        this.classLabels = classLabels;
        
        dataChecks();
    }

    public TimeSeriesInstances(final List<? extends List<? extends List<Double>>> rawData, List<Double> targetValues) {

        int index = 0;
        for (final List<? extends List<Double>> series : rawData) {
            //using the add function means all stats should be correctly counted.
            seriesCollection.add(new TimeSeriesInstance(series, targetValues.get(index++)));
        }

        dataChecks();
    }
    
    public TimeSeriesInstances(final List<? extends List<? extends List<Double>>> rawData, String[] classLabels, final List<Double> labelIndices) {
        this(rawData, labelIndices.stream().map(TimeSeriesInstance::discretiseLabelIndex).collect(Collectors.toList()), classLabels);
    }
    
    public TimeSeriesInstances(final List<? extends List<? extends List<Double>>> rawData, final List<Integer> labelIndexes, String[] classLabels) {
        
        this.classLabels = classLabels;
        
        int index = 0;
        for (final List<? extends List<Double>> series : rawData) {
            //using the add function means all stats should be correctly counted.
            seriesCollection.add(new TimeSeriesInstance(series, labelIndexes.get(index++).intValue(), classLabels));
        }

        dataChecks();
    }

    /**
     * 
     * @param rawData
     * @param labelIndices
     * @param labels
     */
    public TimeSeriesInstances(double[][][] rawData, double[] labelIndices, String[] labels) {
        this(rawData, Arrays.stream(labelIndices).mapToInt(TimeSeriesInstance::discretiseLabelIndex).toArray(), labels);
    }
    
    public TimeSeriesInstances(double[][][] rawData, double[] targetValues) {

        int index = 0;
        for (double[][] series : rawData) {
            //using the add function means all stats should be correctly counted.
            seriesCollection.add(new TimeSeriesInstance(series, targetValues[index++]));
        }
    }
    
    public TimeSeriesInstances(final double[][][] rawData, int[] labelIndexes, String[] labels) {

        classLabels = labels;

        int index = 0;
        for (double[][] series : rawData) {
            //using the add function means all stats should be correctly counted.
            seriesCollection.add(new TimeSeriesInstance(series, labelIndexes[index++], classLabels));
        }

        dataChecks();
    }
	
	public TimeSeriesInstances(List<? extends TimeSeriesInstance> data) {
        this(data, data.isEmpty() ? EMPTY_CLASS_LABELS : data.get(0).getClassLabels());
    }
    
    public TimeSeriesInstances(List<? extends TimeSeriesInstance> data, String[] classLabels) {

        this.classLabels = classLabels;

        seriesCollection.addAll(data);

        dataChecks();
    }
    
    public TimeSeriesInstances(TimeSeriesInstance[] data, String[] classLabels) {
        this(Arrays.asList(data), classLabels);
    }
    
    public TimeSeriesInstances(TimeSeriesInstance[] data) {
        this(Arrays.asList(data));
    }

	private void dataChecks(){
        
        if(seriesCollection == null) {
            throw new NullPointerException("no series collection");
        }
        if(classLabels == null) {
            throw new NullPointerException("no class labels");
        }
        
        calculateLengthBounds();
        calculateIfMissing();
        calculateIfMultivariate();
        calculateNumDimensions();
    }

    private void calculateClassCounts() {
        classCounts = new int[classLabels.length];
        for(TimeSeriesInstance inst : seriesCollection){
            classCounts[inst.getLabelIndex()]++;
        }
    }

    private void calculateLengthBounds() {
        minLength = seriesCollection.stream().mapToInt(TimeSeriesInstance::getMinLength).min().getAsInt();
        maxLength = seriesCollection.stream().mapToInt(TimeSeriesInstance::getMaxLength).max().getAsInt();
        isEqualLength = minLength == maxLength;
    }

    private void calculateNumDimensions(){
        maxNumDimensions = seriesCollection.stream().mapToInt(e -> e.getNumDimensions()).max().getAsInt();
    }
    
    private void calculateIfMultivariate(){
        isMultivariate = seriesCollection.stream().map(TimeSeriesInstance::isMultivariate).anyMatch(Boolean::booleanValue);
    }

    private void calculateIfMissing() {
        // if any of the instance have a missing value then this is true.
        hasMissing = seriesCollection.stream().map(TimeSeriesInstance::hasMissing).anyMatch(Boolean::booleanValue);
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
        calculateClassCounts();
        return classCounts;
    }

    
    /** 
     * @param newSeries
     */
    public void add(final TimeSeriesInstance newSeries) {
        // check that the class labels match
        if(!Arrays.equals(classLabels, newSeries.getClassLabels())) {
            throw new IllegalArgumentException("class labels " + Arrays.toString(classLabels) + " to not match class labels in instance to be added " +
                                                       Arrays.toString(newSeries.getClassLabels()));
        }

        seriesCollection.add(newSeries);

        //guard for if we're going to force update classCounts after.
        if(classCounts != null && newSeries.getLabelIndex() < classCounts.length)
            classCounts[newSeries.getLabelIndex()]++;

        minLength = Math.min(newSeries.getMinLength(), minLength);
        maxLength = Math.max(newSeries.getMaxLength(), maxLength);
        maxNumDimensions = Math.max(newSeries.getNumDimensions(), maxNumDimensions);
        hasMissing |= newSeries.hasMissing();
        isEqualLength = minLength == maxLength;
        isMultivariate |= newSeries.isMultivariate();
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
            out[index++] = inst.getLabelIndex();
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
            out[i++] = inst.getHSliceArray(dim);
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
    
    public Map<Integer, Integer> getHistogramOfLengths(){
        Map<Integer, Integer> out = new TreeMap<>();
        for(TimeSeriesInstance inst : seriesCollection){
            for(TimeSeries ts : inst){
                out.merge(ts.getSeriesLength(), 1, Integer::sum);
            }
        }

        return out;
    }

    @Override public Iterator<TimeSeriesInstance> iterator() {
        return seriesCollection.iterator();
    }

    public Stream<TimeSeriesInstance> stream() {
        return seriesCollection.stream();
    }
    
    public List<List<List<Double>>> getVSliceList(int startInclusive, int endExclusive) {
        return seriesCollection.stream().map(inst -> inst.getVSliceList(startInclusive, endExclusive)).collect(Collectors.toList());
    }

    public TimeSeriesInstances getVSlice(int startInclusive, int endExclusive) {
        final TimeSeriesInstances tsi = new TimeSeriesInstances(classLabels);
        tsi.seriesCollection = seriesCollection.stream().map(inst -> inst.getVSlice(startInclusive, endExclusive)).collect(Collectors.toList());
        tsi.dataChecks();
        return tsi;
    }
    
    public double[][][] getVSliceArray(int startInclusive, int endExclusive) {
        return seriesCollection.stream().map(inst -> inst.getVSliceArray(startInclusive, endExclusive)).toArray(double[][][]::new);
    }
    
    public List<List<List<Double>>> getHSliceList(int startInclusive, int endExclusive) {
        return seriesCollection.stream().map(inst -> inst.getHSliceList(startInclusive, endExclusive)).collect(Collectors.toList());
    }
    
    public double[][][] getHSliceArray(int startInclusive, int endExclusive) {
        return seriesCollection.stream().map(inst -> inst.getHSliceArray(startInclusive, endExclusive)).toArray(double[][][]::new);
    }
    
    public TimeSeriesInstances getHSlice(int startInclusive, int endExclusive) {
        final TimeSeriesInstances tsi = new TimeSeriesInstances(classLabels);
        tsi.seriesCollection = seriesCollection.stream().map(inst -> inst.getHSlice(startInclusive, endExclusive)).collect(Collectors.toList());
        tsi.dataChecks();
        return tsi;
    }

    @Override public boolean equals(final Object o) {
        if(!(o instanceof TimeSeriesInstances)) {
            return false;
        }
        final TimeSeriesInstances that = (TimeSeriesInstances) o;
        return Objects.equals(seriesCollection, that.seriesCollection) && Arrays.equals(classLabels, that.classLabels);
    }

    @Override public int hashCode() {
        return Objects.hash(seriesCollection, classLabels);
    }

    public boolean isClassificationProblem() {
        // if a set of class labels are set then it's a classification problem
        return classLabels.length >= 0;
    }

    public boolean isRegressionProblem() {
        return !isClassificationProblem();
    }
}
