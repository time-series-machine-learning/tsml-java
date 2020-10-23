package tsml.data_containers;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

/**
 * Class to store a time series. The series can have different indices (time stamps) and store missing values (NaN).
 *
 * The model for the indexes is the first is always zero the other indexes are in units of md.increment
 * Hopefully most of this can be encapsulated, so if the data has equal increments then indices is null and the user

 * */
public class TimeSeries extends AbstractList<Double> {

    public static double DEFAULT_VALUE = Double.NaN;

    private List<Double> series;
    private List<Double> indices;


    public TimeSeries(double[] d){
        series = new ArrayList<Double>();
        for(double dd : d)
            series.add(dd);
    }
    
    
    /** 
     * @param ind
     */
    public void setIndices(double[] ind){
        indices = new ArrayList<Double>();
        for(double i : ind)
            indices.add(i);
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
     * Get the length of the series.
     * @return
     */
    @Override public int size() {
        return getSeriesLength();
    }

    @Override public void add(final int i, final Double aDouble) {
        throw new UnsupportedOperationException("time series are not mutable.");
    }

    @Override public Double set(final int i, final Double aDouble) {
        throw new UnsupportedOperationException("time series are not mutable.");
    }

    @Override public void clear() {
        throw new UnsupportedOperationException("time series are not mutable.");
    }

    /** 
     * @return double[]
     */
	public double[] toValuesArray() {
		return getSeries().stream().mapToDouble(Double::doubleValue).toArray();
    }

    
    /** 
     * this is useful if you want to delete a column/truncate the array, but without modifying the original dataset.
     * @param indexesToRemove
     * @return List<Double>
     */
    public List<Double> toListWithoutIndexes(List<Integer> indexesToRemove){
        //if the current index isn't in the removal list, then copy across.
        List<Double> out = new ArrayList<>(this.getSeriesLength() - indexesToRemove.size());
        for(int i=0; i<this.getSeriesLength(); ++i){
            if(!indexesToRemove.contains(i))
                out.add(getOrDefault(i));
        }

        return out;
    }

    
    /** 
     * @param indexesToKeep
     * @return double[]
     */
    public double[] toListWithoutIndexes(int[] indexesToKeep){
        return toArrayWithoutIndexes(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    
    /** 
     * @param indexesToRemove
     * @return double[]
     */
    public double[] toArrayWithoutIndexes(List<Integer> indexesToRemove){
        return toListWithoutIndexes(indexesToRemove).stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    
    /** 
     * this is useful if you want to slice a column/truncate the array, but without modifying the original dataset.
     * @param indexesToKeep
     * @return List<Double>
     */
    public List<Double> toListWithIndexes(List<Integer> indexesToKeep){
        //if the current index isn't in the removal list, then copy across.
        List<Double> out = new ArrayList<>(indexesToKeep.size());
        for(int i=0; i<this.getSeriesLength(); ++i){
            if(indexesToKeep.contains(i))
                out.add(getOrDefault(i));
        }

        return out;
    }

    
    /** 
     * @param indexesToKeep
     * @return double[]
     */
    public double[] toArrayWithIndexes(int[] indexesToKeep){
        return toArrayWithIndexes(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    
    /** 
     * @param indexesToKeep
     * @return double[]
     */
    public double[] toArrayWithIndexes(List<Integer> indexesToKeep){
        return toListWithIndexes(indexesToKeep).stream().mapToDouble(Double::doubleValue).toArray();
    }

    
    /** 
     * @return int
     */
    @Override
    public int hashCode(){
        return this.series.hashCode();
    }


    
    /** 
     * @param args
     */
    public static void main(String[] args) {
        TimeSeries ts = new TimeSeries(new double[]{1,2,3,4}) ;
    }

}
