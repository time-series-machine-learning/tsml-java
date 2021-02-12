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
 
package tsml.examples.ts_examples;

import machine_learning.classifiers.kNN;
import tsml.classifiers.TSClassifier;
import tsml.classifiers.shapelet_based.ShapeletTransformClassifier;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;

public class ClassificationExamples {

    // Using a Weka Classifier the annoying way.
    public static void example1() {
        double[][][] in = {
                // instance zero.
                {
                        // time-series zero.
                        { 0.0, 1.0, 2.0, 4.0, 5.0 }, },
                // instance one
                {
                        // time-series zero.
                        { 4.0, 3.0, 2.0, 1.0 }, } };

        TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[] { 0, 1 }, new String[] { "A", "B" });

        double[][][] in1 = { {
                // time-series zero.
                { 0.0, 1.0, 2.0, 4.0, 5.0 }, } };

        TimeSeriesInstances data2 = new TimeSeriesInstances(in1, new int[] { 0 }, new String[] { "A", "B" });

        kNN nn = new kNN(1);
        nn.buildClassifier(Converter.toArff(data1));

        double acc = ClassifierTools.accuracy(Converter.toArff(data2), nn);
        System.out.println(acc);
    }

    // Using a Weka Classifier the annoying way.
    public static void example2() throws Exception {
        double[][][] in = {
            //instance zero.
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0,5.0},
            },
            //instance one
            {
                //time-series zero.
                {4.0,3.0,2.0,1.0},
            }
        };

        TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[]{0, 1}, new String[] { "A", "B" });

        double[][][] in1 = {   
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0,5.0},
            }
        };

        TimeSeriesInstances data2 = new TimeSeriesInstances(in1, new int[]{0}, new String[] { "A", "B" });

        TSClassifier nn = new TSClassifier(){
            kNN nn = new kNN(1);
			@Override
			public AbstractClassifier getClassifier() {
				return nn;
			}

            public TimeSeriesInstances trainData;
            @Override
            public TimeSeriesInstances getTSTrainData(){
                return trainData;
            }
            
            @Override
            public void setTSTrainData(TimeSeriesInstances train){
                trainData = train;
            }
        };

        nn.buildClassifier(data1);

        double acc = ClassifierTools.accuracy(data2, nn);
        System.out.println(acc);
    }

    // Using a Weka Classifier the annoying way.
    public static void example3() throws Exception {
        double[][][] in = {
            //instance zero.
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0,5.0},
            },
            //instance one
            {
                //time-series zero.
                {4.0,3.0,2.0,1.0, 1.0},
            }
        };

        TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[]{0, 1}, new String[] { "A", "B" });

        double[][][] in1 = {   
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0,5.0},
            }
        };

        TimeSeriesInstances data2 = new TimeSeriesInstances(in1, new int[]{0}, new String[] { "A", "B" });

        TSClassifier stc = new ShapeletTransformClassifier();
        stc.buildClassifier(data1);
        double acc = ClassifierTools.accuracy(data2, stc);
        System.out.println(acc);
    }


    public static void main(String[] args) throws Exception {
        example2();
    }


}
