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

        TimeSeriesInstances newTrain = new TimeSeriesInstances(new_train, train.getClassLabels());
        TimeSeriesInstances newTest = new TimeSeriesInstances(new_test, test.getClassLabels());

        // set problem name
        newTrain.setProblemName(train.getProblemName());
        newTest.setProblemName(test.getProblemName());

        // set description
        newTrain.setDescription(train.getDescription());
        newTest.setDescription(test.getDescription());

        // set class counts
        train.getClassCounts();
        test.getClassCounts();

        return new TrainTest(newTrain, newTest);
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