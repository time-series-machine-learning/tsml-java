package tsml.data_containers;

import java.util.ArrayList;
import java.util.List;

/**
 * Data structure able to store a time series instance.
 * it can be standard (univariate, no missing, equally sampled series) or
 * complex (multivariate, unequal length, unequally spaced, univariate or multivariate time series).
 *
 * Should Instances be immutable after creation? Meta data is calculated on creation, mutability can break this
 */

public class TimeSeriesInstance {

    //this ctor can be made way more sophisticated.
    public TimeSeriesInstance(List<List<Double>> series, Double label) {
        this(series);

        classLabel = label.intValue();
    }


    public TimeSeriesInstance(List<List<Double>> series){
        //process the input list to produce TimeSeries Objects.
        //this allows us to pad if need be, or if we want to squarify the data etc.
        series_channels = new ArrayList<TimeSeries>();

        for(List<Double> channel : series){
            //convert List<Double> to double[]
            series_channels.add(new TimeSeries(channel.stream().mapToDouble(Double::doubleValue).toArray()));
        }
    }
    
	List<TimeSeries> series_channels;
    Integer classLabel;

    public int getNumChannels(){
        return series_channels.size();
    }


    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();

        sb.append("Num Channels: ").append(getNumChannels()).append(" Class Label: ").append(classLabel);
        for(TimeSeries channel : series_channels){
            sb.append(System.lineSeparator());
            sb.append(channel.toString());
        }

        return sb.toString();
    }


}