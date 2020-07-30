package tsml.transformers;

import java.util.ArrayList;
import java.util.List;

import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;


//This class if for weird hacky dimension wise operations we need to do when interfacing with Weka classifiers
//that can only take univeraite data.
public class Splitter{

    //splitty splitty.
    public static List<TimeSeriesInstance> SplitTimeSeriesInstance(TimeSeriesInstance inst){
        List<TimeSeriesInstance> output = new ArrayList<>(inst.getNumChannels());

        for(TimeSeries ts : inst){
            double[][] wrapped_raw = new double[1][];
            wrapped_raw[0] = ts.toArray();

            output.add(new TimeSeriesInstance(wrapped_raw, inst.getLabelIndex()));
        }

        return output;
    }

    //horizontally slice into univariate TimeSeriesInstances.
    public static List<TimeSeriesInstances> SplitTimeSeriesInstances(TimeSeriesInstances inst){
        List<TimeSeriesInstances> output = new ArrayList<>(inst.getMaxNumChannels());

        for(int i=0; i< inst.getMaxNumChannels(); i++){
            TimeSeriesInstances temp = new TimeSeriesInstances(inst.getHSliceArray(new int[]{i}), inst.getClassIndexes());
            temp.setClassLabels(inst.getClassLabels());
            output.add(temp);
        }

        return output;
    }

    //mergey mergey
    public static TimeSeriesInstance MergeTimeSeriesInstance(List<TimeSeriesInstance> inst_dims){
        double[][] wrapped_raw = new double[inst_dims.size()][];
        int i=0;
        for(TimeSeriesInstance inst : inst_dims){
            //ignore any other dimensions, because they should only be single 
            wrapped_raw[i++] = inst.toValueArray()[0]; 
        }   

        return new TimeSeriesInstance(wrapped_raw, inst_dims.get(0).getLabelIndex());
    }
    
}