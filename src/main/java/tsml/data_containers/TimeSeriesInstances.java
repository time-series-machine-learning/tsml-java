package tsml.data_containers;

import java.util.ArrayList;
import java.util.List;

/**
 * Data structure able to handle unequal length, unequally spaced, univariate or
 * multivariate time series.
 */
public class TimeSeriesInstances {


    /* Meta Information  */
    boolean isEquallySpaced;
    boolean hasMissing;
    boolean isEqualLength;

    //this could be by dimension, so could be a list.
    int minLength;
    int maxLength;

    public boolean hasMissing(){
        return hasMissing;
    }

    public boolean isEuqallySpaced(){
        return isEquallySpaced;
    }

    public boolean isEqualLength(){
        return isEqualLength;
    }

    public int getMinLength(){
        return minLength;
    }

    public int getMaxLength(){
        return maxLength;
    }

    /* End Meta Information  */



    List<TimeSeriesInstance> series_collection;

    //mapping for class labels. so ["apple","orange"] => [0,1]
    //this could be optional for example regression problems.
    List<String> classLabels;

    public TimeSeriesInstances(){
        series_collection = new ArrayList<>();
    }

    public TimeSeriesInstances(List<List<List<Double>>> raw_data, List<Double> label_indexes){
        this();
        
        int index = 0;
        for(List<List<Double>> series : raw_data){
            series_collection.add(new TimeSeriesInstance(series, label_indexes.get(index++)));
        }

        calculateLengthBounds();
        calculateIfMissing();
    }

    private void calculateLengthBounds(){
        minLength = series_collection.stream().mapToInt(e -> e.minLength).min().getAsInt();
        maxLength = series_collection.stream().mapToInt(e -> e.maxLength).max().getAsInt();
        isEqualLength = minLength == maxLength;
    }

    private void calculateIfMissing(){
        //if any of the instance have a missing value then this is true.
        hasMissing = series_collection.stream().map(e ->  e.hasMissing).anyMatch(Boolean::booleanValue);
    }
    
    public TimeSeriesInstances(List<List<List<Double>>> raw_data){
        this();
        
        for(List<List<Double>> series : raw_data){
            series_collection.add(new TimeSeriesInstance(series));
        }
    }

    public void setClassLabels(List<String> labels){
        classLabels = labels;
    }


	public void add(TimeSeriesInstance new_series) {
        series_collection.add(new_series);
    }
    
    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();

        sb.append("Labels: [").append(classLabels.get(0));
        for(int i=1; i<classLabels.size(); i++){
            sb.append(',');
            sb.append(classLabels.get(i));
        }
        sb.append(']').append(System.lineSeparator());

        for(TimeSeriesInstance series : series_collection){
            sb.append(series.toString());
            sb.append(System.lineSeparator());
        }

        return sb.toString();
    }
}
