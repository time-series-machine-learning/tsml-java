package tsml.data_containers;

import java.io.File;
import java.util.ArrayList;
import java.util.Date;

/**
 * Class to store a time series. The series can have different indices (time stamps) and store missing values (NaN).
 *
 * The model for the indexes is the first is always zero the other indexes are in units of md.increment
 * Hopefully most of this can be encapsulated, so if the data has equal increments then indices is null and the user

 * */
public class TimeSeries {
    private double[] series;
    private double[] indices;
    MetaData md;


    public TimeSeries(double[] d){
        series = d;
    }
    
	public void setSeries(double[] d){
        series=d;
    }
    public void setIndices(double[] ind){
        indices=ind;
    }
    public void setSeriesAndIndex(double[] d,double[] ind){
        series=d;
        indices=ind;
    }
    public double[] getSeries(){ return series;}
    public double[] getIndices(){ return indices;}

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
}