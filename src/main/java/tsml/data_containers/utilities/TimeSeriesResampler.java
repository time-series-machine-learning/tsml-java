package tsml.data_containers.utilities;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import weka.core.Instances;

public class TimeSeriesResampler {
    

    public static TrainTest resampleTrainTest(TimeSeriesInstances train, TimeSeriesInstances test, long seed){

        if(seed == 0)
            return new TrainTest(train, test);

        //create combined list.
        List<TimeSeriesInstance> all = new ArrayList<>(train.numInstances() + test.numInstances());
        all.addAll(train.getAll());
        all.addAll(test.getAll());

        int[] classCounts = train.getClassCounts();

        //build the map.
        Map<Integer, List<TimeSeriesInstance>> classBins = new HashMap<>();
        for(TimeSeriesInstance inst : all){
            List<TimeSeriesInstance> values = classBins.computeIfAbsent(inst.getLabelIndex(), k -> new ArrayList<>());
            values.add(inst);
        }

        Random r = new Random(seed);

        List<TimeSeriesInstance> new_train = new ArrayList<>();
        List<TimeSeriesInstance> new_test = new ArrayList<>();
        for(Integer classVal : classBins.keySet()){
            int occurences = classCounts[classVal.intValue()];
            List<TimeSeriesInstance> bin = classBins.get(classVal);
            randomize(bin,r); //randomise the bin.

            new_train.addAll(bin.subList(0,occurences));//copy the first portion of the bin into the train set
            new_test.addAll(bin.subList(occurences, bin.size()));//copy the remaining portion of the bin into the test set.
        }

        return new TrainTest(new TimeSeriesInstances(new_train, train.getClassLabels()), 
                             new TimeSeriesInstances(new_test, train.getClassLabels()));
    }

    //this function is the one from Instances, want to mirror there shuffling algorithm.
    private static void randomize(List<TimeSeriesInstance> data, Random random) {
        for (int j = data.size() - 1; j > 0; j--)
            swap(data, j, random.nextInt(j+1));
    }

    //this function is the same as 
    private static void swap(List<TimeSeriesInstance> data, int i, int j){
        TimeSeriesInstance in = data.get(i);
        data.set(i, data.get(j));
        data.set(j, in);
    }
      

    public static class TrainTest{
        public TrainTest(TimeSeriesInstances train, TimeSeriesInstances test) {
            this.train = train;
            this.test = test;
        }

        public TimeSeriesInstances train;
        public TimeSeriesInstances test;
    }
}