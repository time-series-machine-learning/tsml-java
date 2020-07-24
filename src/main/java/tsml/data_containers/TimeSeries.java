package tsml.data_containers;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * Class to store a time series. The series can have different indices (time stamps) and store missing values (NaN).
 *
 * The model for the indexes is the first is always zero the other indexes are in units of md.increment
 * Hopefully most of this can be encapsulated, so if the data has equal increments then indices is null and the user

 * */
public class TimeSeries{

    /*
    private double[] series;
    private double[] indices;
    */

    private List<Double> series;
    private List<Double> indices;
    MetaData md;


    public TimeSeries(double[] d){
        series = new ArrayList<Double>();
        for(double dd : d)
            series.add(dd);
    }
    
    public void setIndices(double[] ind){
        indices = new ArrayList<Double>();
        for(double i : ind)
            indices.add(i);
    }

    public int getSeriesLength(){
        return series.size();
    }

    public boolean hasValidValueAt(int i){
        //test whether its out of range, or NaN
        boolean output = i < series.size() &&
                         Double.isFinite(series.get(i));
        return output;
    }

    public double get(int i){
        return series.get(i);
    }

    public DoubleStream stream(){
        return series.stream().mapToDouble(Double::doubleValue);
    }

    public List<Double> getSlidingWindow(int start, int end){
        return series.subList(start, end);
    }

    public double[] getSlidingWindowArray(int start, int end){
        return series.subList(start, end).stream().mapToDouble(Double::doubleValue).toArray();
    }

    // public void setSeriesAndIndex(double[] d,double[] ind){
    //     series=d;
    //     indices=ind;
    // }
    public List<Double> getSeries(){ return series;}
    public List<Double> getIndices(){ return indices;}

    private class MetaData{
        String name;
        Date startDate;
        double increment;  //Base unit to be ....... 1 day?

    }

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();

        for(double val : series){
            sb.append(val).append(',');
        }

        return sb.toString();
    }

    //TODO: Should this filter out NaNs.
    //TODO: Or Should we have a second toArray which is toArraySafe
	public double[] toArray() {
		return getSeries().stream().mapToDouble(Double::doubleValue).toArray();
    }

    //this is useful if you want to delete a column/truncate the array, but without modifying the original dataset.
    public List<Double> toListWithoutIndexes(List<Integer> indexesToRemove){
        //if the current index isn't in the removal list, then copy across.
        List<Double> out = new ArrayList<>(this.getSeriesLength() - indexesToRemove.size());
        for(int i=0; i<this.getSeriesLength(); ++i){
            if(!indexesToRemove.contains(i))
                out.add(this.series.get(i));
        }

        return out;
    }

    public double[] toListWithoutIndexes(int[] indexesToKeep){
        return toArrayWithoutIndexes(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    public double[] toArrayWithoutIndexes(List<Integer> indexesToRemove){
        return toListWithoutIndexes(indexesToRemove).stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    //this is useful if you want to slice a column/truncate the array, but without modifying the original dataset.
    public List<Double> toListWithIndexes(List<Integer> indexesToKeep){
        //if the current index isn't in the removal list, then copy across.
        List<Double> out = new ArrayList<>(indexesToKeep.size());
        for(int i=0; i<this.getSeriesLength(); ++i){
            if(indexesToKeep.contains(i))
                out.add(this.series.get(i));
        }

        return out;
    }

    public double[] toArrayWithIndexes(int[] indexesToKeep){
        return toArrayWithIndexes(Arrays.stream(indexesToKeep).boxed().collect(Collectors.toList()));
    }

    public double[] toArrayWithIndexes(List<Integer> indexesToKeep){
        return toListWithIndexes(indexesToKeep).stream().mapToDouble(Double::doubleValue).toArray();
    }

    public static void main(String[] args) {
        TimeSeries ts = new TimeSeries(new double[]{1,2,3,4}) ;
    }


}