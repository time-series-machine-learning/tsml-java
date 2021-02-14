/* 
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.data_containers;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Data structure able to store a time series instance. it can be standard
 * (univariate, no missing, equally sampled series) or complex (multivariate,
 * unequal length, unequally spaced, univariate or multivariate time series).
 *
 * Should Instances be immutable after creation? Meta data is calculated on
 * creation, mutability can break this
 *
 @author Aaron Bostrom, 2020
 */

public class TimeSeriesInstance implements Iterable<TimeSeries> {

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
    private List<TimeSeries> seriesDimensions;
    private int labelIndex = -1;
    private double targetValue = Double.NaN;
    
    public TimeSeriesInstance(double targetValue, List<? extends TimeSeries> series) {
        this.seriesDimensions = new ArrayList<>(series);
        this.targetValue = targetValue;
        
        dataChecks();
    }
    
    public TimeSeriesInstance(int labelIndex, List<? extends TimeSeries> series) {
        this.seriesDimensions = new ArrayList<>(series);
        this.labelIndex = labelIndex;
        
        dataChecks();
    }

    /**
     * Construct a labelled instance from raw data.
     * @param series
     * @param label
     * @param classLabels
     */
    public TimeSeriesInstance(List<? extends List<Double>> series, int label) {
        this(series, Double.NaN);

        targetValue = labelIndex = label;
        
        dataChecks();
    }
    
    public TimeSeriesInstance(List<? extends List<Double>> series, double targetValue) {
        // process the input list to produce TimeSeries Objects.
        // this allows us to pad if need be, or if we want to squarify the data etc.
        seriesDimensions = new ArrayList<TimeSeries>();

        for (List<Double> ts : series) {
            seriesDimensions.add(new TimeSeries(ts));
        }
        
        this.targetValue = targetValue;

        dataChecks();
    }

    /**
     * Construct an regressed instance from raw data.
     * @param data
     * @param targetValue
     */
	public TimeSeriesInstance(double[][] data, double targetValue) {
        seriesDimensions = new ArrayList<TimeSeries>();

        for(double[] in : data){
            seriesDimensions.add(new TimeSeries(in));
        }
        
        this.targetValue = targetValue;

        dataChecks();
    }

    /**
     * Construct an labelled instance from raw data.
     * @param data
     * @param labelIndex
     * @param classLabels
     */
    public TimeSeriesInstance(double[][] data, int labelIndex, String[] classLabels) {
        seriesDimensions = new ArrayList<TimeSeries>();

        for(double[] in : data){
            seriesDimensions.add(new TimeSeries(in));
        }

        targetValue = this.labelIndex = labelIndex;
        
        dataChecks();
    }

        /**
     * Construct an regressed instance from raw data.
     * @param data
     * @param targetValue
     */
	public TimeSeriesInstance(double[][] data, int labelIndex) {
        seriesDimensions = new ArrayList<TimeSeries>();

        for(double[] in : data){
            seriesDimensions.add(new TimeSeries(in));
        }
        System.out.println(labelIndex);
        this.labelIndex = labelIndex;

        dataChecks();
    }
    
    public static int discretiseLabelIndex(double labelIndex) {
        final int i;
        if(Double.isNaN(labelIndex)) {
            i = -1;
        } else {
            i = (int) labelIndex;
            // check the given double is an integer, i.e. 3.0 == 3. Protects against abuse through implicit label indexing integer casting, i.e. 3.3 --> 3. The user should do this themselves, otherwise it's safest to assume a non-integer value (e.g. 7.4) is an error and raise exception.
            if(labelIndex != i) {
                throw new IllegalArgumentException("cannot discretise " + labelIndex + " to an int: " + i);
            }
        }
        return i;
    }

    /**
     * Construct a labelled instance from raw data with label in double form (but should be an integer value).
     * @param data
     * @param labelIndex
     * @param classLabels
     */
    public TimeSeriesInstance(double[][] data, double labelIndex, String[] classLabels) {
        this(data, discretiseLabelIndex(labelIndex), classLabels);
    }

    /**
     * Construct an instance from raw data. Copies over regression target / labelling variables. This is only intended for internal use in avoiding copying the data again after a vslice / hslice.
     * @param data
     * @param other
     */
    private TimeSeriesInstance(double[][] data, TimeSeriesInstance other) {
        this(data, Double.NaN);
        labelIndex = other.labelIndex;
        targetValue = other.targetValue;
        
        dataChecks();
    }
    
    public TimeSeriesInstance(double[][] data) {
        this(data, Double.NaN);
    }
    
    public TimeSeriesInstance(List<? extends List<Double>> data) {
        this(data, Double.NaN);
    }
    
    private TimeSeriesInstance() {}
    
    private TimeSeriesInstance copy() {
        final TimeSeriesInstance inst = new TimeSeriesInstance();
        inst.labelIndex = labelIndex;
        inst.seriesDimensions = seriesDimensions;
        inst.targetValue = targetValue;
        
        inst.dataChecks();
        
        return inst;
    }
    
    public TimeSeriesInstance(double targetValue, TimeSeries[] data) {
        this(targetValue, Arrays.asList(data));
    }
    
    public TimeSeriesInstance(int labelIndex, TimeSeries[] data) {
        this(labelIndex, Arrays.asList(data));
    }


    private void dataChecks(){
        
        if(seriesDimensions == null) {
            throw new NullPointerException("no series dimensions");
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
        return new TimeSeriesInstance(getVSliceArray(indexesToKeep), this);
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
        return new TimeSeriesInstance(getHSliceArray(dimensionsToKeep), this);
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

    @Override public Iterator<TimeSeries> iterator() {
        return seriesDimensions.iterator();
    }
    
    public Stream<TimeSeries> stream() {
        return seriesDimensions.stream();
    }
    
    public TimeSeriesInstance getHSlice(int startInclusive, int endExclusive) {
        // copy construct a new inst
        final TimeSeriesInstance tsi = copy();
        // trim current data to a subset
        tsi.seriesDimensions = seriesDimensions.subList(startInclusive, endExclusive);
        tsi.dataChecks();
        return tsi;
    }
    
    public List<List<Double>> getHSliceList(int startInclusive, int endExclusive) {
        return seriesDimensions.subList(startInclusive, endExclusive).stream().map(TimeSeries::getSeries).collect(Collectors.toList());
    }
    
    public double[][] getHSliceArray(int startInclusive, int endExclusive) {
        return getHSliceList(startInclusive, endExclusive).stream().map(dim -> dim.stream().mapToDouble(d -> d).toArray()).toArray(double[][]::new);
    }
    
    public List<List<Double>> getVSliceList(int startInclusive, int endExclusive) {
        return seriesDimensions.stream().map(dim -> dim.getVSliceList(startInclusive, endExclusive)).collect(Collectors.toList());
    }
    
    public double[][] getVSliceArray(int startInclusive, int endExclusive) {
        return getVSliceList(startInclusive, endExclusive).stream().map(dim -> dim.stream().mapToDouble(d -> d).toArray()).toArray(double[][]::new);
    }
    
    public TimeSeriesInstance getVSlice(int startInclusive, int endExclusive) {
        // copy construct a new inst
        final TimeSeriesInstance tsi = copy();
        // trim current data to a subset
        tsi.seriesDimensions = seriesDimensions.stream().map(dim -> dim.getVSlice(startInclusive, endExclusive)).collect(Collectors.toList());
        tsi.dataChecks();
        return tsi;
    }

    @Override public boolean equals(final Object o) {
        if(!(o instanceof TimeSeriesInstance)) {
            return false;
        }
        final TimeSeriesInstance that = (TimeSeriesInstance) o;
        return labelIndex == that.labelIndex &&
                       Double.compare(that.targetValue, targetValue) == 0 &&
                       seriesDimensions.equals(that.seriesDimensions);
    }

    @Override public int hashCode() {
        
        return Objects.hash(seriesDimensions, labelIndex);
    }
    
    public boolean isLabelled() {
        // is labelled if label index points to a class label
        return labelIndex >= 0;
    }
    
    public boolean isRegressed() {
        // is regressed if the target value is set
        return targetValue != Double.NaN;
    }
    
    public boolean isClassificationProblem() {
        // if a set of class labels are set then it's a classification problem
        return labelIndex >= 0;
    }
    
    public boolean isRegressionProblem() {
        return !isClassificationProblem();
    }
}
