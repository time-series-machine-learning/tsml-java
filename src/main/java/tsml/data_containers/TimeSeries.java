package tsml.data_containers;

import java.util.Date;

/**
 * Class to store a time series. The series can have different indices (time stamps) and store missing values (NaN).

 * */
public class TimeSeries {
    double[] series;
    double[] indices;
    MetaData md;

    public void setSeries(double[] d){
        series=d;
    }
    public void setIndex(double[] ind){
        indices=ind;
    }

    public class MetaData{
        String name;
        Date startDate;
        double increment;  //Base unit to be ....... 1 day?

    }
}