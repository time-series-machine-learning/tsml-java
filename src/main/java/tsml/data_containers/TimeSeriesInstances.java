package tsml.data_containers;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Data structure able to handle unequal length, unequally spaced, univariate or
 * multivariate time series.
 */
public class TimeSeriesInstances extends AbstractList<TimeSeriesInstance> {

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
        return classLabels.size();
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

    private List<TimeSeriesInstance> seriesCollection;

    // mapping for class labels. so ["apple","orange"] => [0,1]
    // this could be optional for example regression problems.
    private List<String> classLabels;

    private int[] classCounts;

    /**
     * Build from instances and known class labels set.
     * @param instances
     * @param classLabels
     */
    public TimeSeriesInstances(List<TimeSeriesInstance> instances, List<String> classLabels) {
        this.seriesCollection = new ArrayList<>(instances);
        this.classLabels = Collections.unmodifiableList(classLabels);
        dataChecks();
    }

    public TimeSeriesInstances(List<String> classLabels) {
        this(new ArrayList<>(), classLabels);
    }
    
    private void dataChecks() {
        // check all the class labels line up
        // for regressed instances, the class labels should be an empty list
        for(TimeSeriesInstance instance : this) {
            List<String> classLabelsInInstance = instance.getClassLabels();
            if(!classLabels.equals(classLabelsInInstance)) {
                throw new IllegalArgumentException("labels mismatch: " + classLabelsInInstance + " do not match " + classLabels);
            }
        }

        calculateLengthBounds();
        calculateIfMissing();
        calculateIfMultivariate();
        calculateNumDimensions();

    }
    
    public static TimeSeriesInstances fromData(double[][][] data) {
        final ArrayList<TimeSeriesInstance> insts = new ArrayList<>(data.length);
        for(double[][] instData : data) {
            TimeSeriesInstance inst = TimeSeriesInstance.fromData(instData);
            insts.add(inst);
        }
        return new TimeSeriesInstances(insts);
    }

    public static TimeSeriesInstances fromLabelledData(double[][][] data, int[] labels, List<String> classLabels) {
        final ArrayList<TimeSeriesInstance> insts = new ArrayList<>(data.length);
        classLabels = Collections.unmodifiableList(classLabels);
        for(int i = 0; i < data.length; i++) {
            final double[][] instData = data[i];
            final int label = labels[i];
            TimeSeriesInstance inst = TimeSeriesInstance.fromLabelledData(instData, label, classLabels);
            insts.add(inst);
        }
        return new TimeSeriesInstances(insts);
    }
    
    public static TimeSeriesInstances fromLabelledDataDouble(double[][][] data, double[] labels, List<String> classLabels) {
        int[] labelIndices = new int[labels.length];
        for(int j = 0; j < labels.length; j++) {
            final double d = labels[j];
            int i = (int) d;
            if(i != d) {
                throw new IllegalArgumentException("non-discrete label indices: " + d);
            }
            labelIndices[j] = i;
        }
        return fromLabelledData(data, labelIndices, classLabels);
    }
    
    public static TimeSeriesInstances fromRegressedData(double[][][] data, double[] targetValues) {
        final ArrayList<TimeSeriesInstance> insts = new ArrayList<>(data.length);
        for(int i = 0; i < data.length; i++) {
            final double[][] instData = data[i];
            double targetValue = targetValues[i];
            TimeSeriesInstance inst = TimeSeriesInstance.fromRegressedData(instData, targetValue);
            insts.add(inst);
        }
        return new TimeSeriesInstances(insts);
    }
    
    public static TimeSeriesInstances fromData(List<List<List<Double>>> data) {
        final ArrayList<TimeSeriesInstance> insts = new ArrayList<>(data.size());
        for(List<List<Double>> instData : data) {
            TimeSeriesInstance inst = TimeSeriesInstance.fromData(instData);
            insts.add(inst);
        }
        return new TimeSeriesInstances(insts);
    }

    public static TimeSeriesInstances fromLabelledData(List<List<List<Double>>> data, List<Integer> labels, List<String> classLabels) {
        final ArrayList<TimeSeriesInstance> insts = new ArrayList<>(data.size());
        classLabels = Collections.unmodifiableList(classLabels);
        for(int i = 0; i < data.size(); i++) {
            final List<List<Double>> instData = data.get(i);
            final int label = labels.get(i);
            TimeSeriesInstance inst = TimeSeriesInstance.fromLabelledData(instData, label, classLabels);
            insts.add(inst);
        }
        return new TimeSeriesInstances(insts);
    }

    public static TimeSeriesInstances fromLabelledDataDouble(List<List<List<Double>>> data, List<Double> labels, List<String> classLabels) {
        final ArrayList<TimeSeriesInstance> insts = new ArrayList<>(data.size());
        classLabels = Collections.unmodifiableList(classLabels);
        for(int i = 0; i < data.size(); i++) {
            final List<List<Double>> instData = data.get(i);
            Double labelDouble = labels.get(i);
            final int label = labelDouble.intValue();
            if(labelDouble != label) {
                throw new IllegalArgumentException("non-discrete label: " + labelDouble);
            }
            TimeSeriesInstance inst = TimeSeriesInstance.fromLabelledData(instData, label, classLabels);
            insts.add(inst);
        }
        return new TimeSeriesInstances(insts);
    }

    public static TimeSeriesInstances fromRegressedData(List<List<List<Double>>> data, List<Double> targetValues) {
        final ArrayList<TimeSeriesInstance> insts = new ArrayList<>(data.size());
        for(int i = 0; i < data.size(); i++) {
            final List<List<Double>> instData = data.get(i);
            final double targetValue = targetValues.get(i);
            TimeSeriesInstance inst = TimeSeriesInstance.fromRegressedData(instData, targetValue);
            insts.add(inst);
        }
        return new TimeSeriesInstances(insts);
    }

    private void calculateClassCounts() {
        classCounts = new int[classLabels.size()];
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
    public List<String> getClassLabels() {
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

    public String[] getClassLabelsArray() {
        return classLabels.toArray(new String[0]);
    }
    
    /** 
     * @param newSeries
     */
    public void add(int i, final TimeSeriesInstance newSeries) {
        seriesCollection.add(i, newSeries);

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


        sb.append("Labels: ").append(classLabels).append(System.lineSeparator());

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

    @Override public TimeSeriesInstance set(final int i, final TimeSeriesInstance instance) {
        throw new UnsupportedOperationException("TimeSeriesInstances not mutable");
    }

    @Override public TimeSeriesInstance remove(final int i) {
        throw new UnsupportedOperationException("TimeSeriesInstances not mutable");
    }

    @Override public void clear() {
        throw new UnsupportedOperationException("TimeSeriesInstances not mutable");
    }

    @Override public int size() {
        return numInstances();
    }
}
