package tsml.data_containers;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * Class to store a time series. The series can have different indices (time stamps) and store missing values (NaN).
 *
 * The model for the indexes is the first is always zero the other indexes are in units of md.increment
 * Hopefully most of this can be encapsulated, so if the data has equal increments then indices is null and the user

 * */
public class TimeSeries implements Iterable<Double> {

    public final static double DEFAULT_VALUE = Double.NaN;
    private final static List<Double> EMPTY_INDICES = Collections.emptyList(); 

    private List<Double> series;
    private List<Double> indices = EMPTY_INDICES;

    private TimeSeries() {
        // just for internal use
    }
    
    public TimeSeries(double[] d){
        series = new ArrayList<Double>();
        for(double dd : d)
            series.add(dd);
    }
    
    public TimeSeries(List<Double> d) {
        series = new ArrayList<>(d);
    }
    
    public TimeSeries(TimeSeries other) {
        this(other.series);
    }

    
    /** 
     * @return int
     */
    public int getSeriesLength(){
        return series.size();
    }

    
    /** 
     * @param i
     * @return boolean
     */
    public boolean hasValidValueAt(int i){
        //test whether its out of range, or NaN
        boolean output = i < series.size() &&
                         Double.isFinite(series.get(i));
        return output;
    }

    
    /** 
     * @param i
     * @return double
     */
    public double getValue(int i){
        return series.get(i);
    }

    /**
     * Gets a value at a specific index in the time series. This method conducts unboxing so use getValue if you care about performance.
     * @param i
     * @return
     */
    public Double get(int i) {
        return series.get(i);
    }
    
    /** 
     * @param i
     * @return double
     */
    public double getOrDefault(int i){
        return hasValidValueAt(i) ? getValue(i) : DEFAULT_VALUE;
    }

    
    /** 
     * @return DoubleStream
     */
    public DoubleStream streamValues(){
        return series.stream().mapToDouble(Double::doubleValue);
    }
    
    public Stream<Double> stream() {
        return series.stream();
    }
    
    /** 
     * @param start
     * @param end
     * @return List<Double>
     */
    public List<Double> getSlidingWindow(int start, int end){
        return series.subList(start, end);
    }

    
    /** 
     * @param start
     * @param end
     * @return double[]
     */
    public double[] getSlidingWindowArray(int start, int end){
        return series.subList(start, end).stream().mapToDouble(Double::doubleValue).toArray();
    }

    
    /** 
     * @return List<Double>
     */
    public List<Double> getSeries(){ return series;}
    
    /** 
     * @return List<Double>
     */
    public List<Double> getIndices(){ return indices;}

    
    /** 
     * @return String
     */
    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();

        for(double val : series){
            sb.append(val).append(',');
        }

        return sb.toString();
    }

    /** 
     * @return double[]
     */
	public double[] toValueArray() {
		return getSeries().stream().mapToDouble(Double::doubleValue).toArray();
    }

    public TimeSeries getVSlice(int[] indices) {
        return new TimeSeries(getVSliceArray(indices));
    }
    
    public TimeSeries getVSlice(int index) {
	    return getVSlice(new int[] {index});
    }

    public TimeSeries getVSlice(List<Integer> indices) {
	    return getVSlice(indices.stream().mapToInt(Integer::intValue).toArray());
    }

    public TimeSeries getVSliceComplement(int index) {
	    return getVSliceComplement(new int[] {index});
    }
    
    public TimeSeries getVSliceComplement(int[] indices) {
        return new TimeSeries(getVSliceComplementArray(indices));
    }

    public TimeSeries getVSliceComplement(List<Integer> indices) {
        return getVSliceComplement(indices.stream().mapToInt(Integer::intValue).toArray());
    }
    
    /** 
     * this is useful if you want to delete a column/truncate the array, but without modifying the original dataset.
     * @param indexesToRemove
     * @return List<Double>
     */
    public List<Double> getVSliceComplementList(List<Integer> indexesToRemove){
        //if the current index isn't in the removal list, then copy across.
        List<Double> out = new ArrayList<>(this.getSeriesLength() - indexesToRemove.size());
        for(int i=0; i<this.getSeriesLength(); ++i){
            if(!indexesToRemove.contains(i))
                out.add(getOrDefault(i));
        }

        return out;
    }
    
    public List<Double> getVSliceComplementList(int[] indexesToRemove) {
        return getVSliceComplementList(Arrays.stream(indexesToRemove).boxed().collect(Collectors.toList()));
    }

    public List<Double> getVSliceComponentList(int index) {
        return getVSliceComplementList(new int[] {index});
    }
    
    /** 
     * @param indexesToKeep
     * @return double[]
     */
    public double[] getVSliceComplementArray(int[] indexesToKeep){
        return getVSliceComplementArray(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    
    /** 
     * @param indexesToRemove
     * @return double[]
     */
    public double[] getVSliceComplementArray(List<Integer> indexesToRemove){
        return getVSliceComplementList(indexesToRemove).stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    public double[] getVSliceComplementArray(int index) {
        return getVSliceComplementArray(new int[] {index});
    }
    
    /** 
     * this is useful if you want to slice a column/truncate the array, but without modifying the original dataset.
     * @param indexesToKeep
     * @return List<Double>
     */
    public List<Double> getVSliceList(List<Integer> indexesToKeep){
        //if the current index isn't in the removal list, then copy across.
        List<Double> out = new ArrayList<>(indexesToKeep.size());
        for(int i=0; i<this.getSeriesLength(); ++i){
            if(indexesToKeep.contains(i))
                out.add(getOrDefault(i));
        }

        return out;
    }
    
    public List<Double> getVSliceList(int[] indexesToKeep) {
        return getVSliceList(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }
    
    public List<Double> getVSliceList(int index) {
        return getVSliceList(new int[] {index});
    }

    public double[] getVSliceArray(int index) {
        return getVSliceArray(new int[] {index});
    }
    
    /** 
     * @param indexesToKeep
     * @return double[]
     */
    public double[] getVSliceArray(int[] indexesToKeep){
        return getVSliceArray(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    
    /** 
     * @param indexesToKeep
     * @return double[]
     */
    public double[] getVSliceArray(List<Integer> indexesToKeep){
        return getVSliceList(indexesToKeep).stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    /** 
     * @param args
     */
    public static void main(String[] args) {
        TimeSeries ts = new TimeSeries(new double[]{1,2,3,4}) ;
    }

    @Override public Iterator<Double> iterator() {
        return series.iterator();
    }
    
    public List<Double> getVSliceList(int startInclusive, int endExclusive) {
        return series.subList(startInclusive, endExclusive);
    }
    
    public double[] getVSliceArray(int startInclusive, int endExclusive) {
        return getVSliceList(startInclusive, endExclusive).stream().mapToDouble(d -> d).toArray();
    }
    
    public TimeSeries getVSlice(int startInclusive, int endExclusive) {
        final TimeSeries ts = new TimeSeries();
        ts.series = getVSliceList(startInclusive, endExclusive);
        return ts;
    }

    @Override public boolean equals(final Object o) {
        if(!(o instanceof TimeSeries)) {
            return false;
        }
        final TimeSeries that = (TimeSeries) o;
        return Objects.equals(series, that.series);
    }

    @Override public int hashCode() {
        return Objects.hash(series);
    }
}
