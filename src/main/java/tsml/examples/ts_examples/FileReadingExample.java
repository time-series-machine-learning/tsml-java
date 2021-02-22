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

import tsml.classifiers.interval_based.TSF;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.ts_fileIO.TSReader;
import tsml.data_containers.utilities.Converter;
import utilities.ClassifierTools;
import weka.core.Instances;

import java.io.*;

/**
 * Example to show how to read in .ts and .arff data to the new data model
 * (TimeSeriesInstances, TimeSeriesInstance, TimeSeries)
 *
 * @author Conor Egan (c-eg)
 */
public class FileReadingExample {
    public static void main(String[] args) throws Exception {
        tsfWithARFF();
        tsfWithTS();
    }

    /**
     * Load data with .ts data files to build TSF classifier
     */
    private static void tsfWithTS() throws Exception {
        // data locations, change this to point to your datasets
        String trainDataTS = "D:\\Documents\\Project stuff\\Datasets\\ItalyPowerDemand\\ItalyPowerDemand_TRAIN.ts";
        String testDataTS = "D:\\Documents\\Project stuff\\Datasets\\ItalyPowerDemand\\ItalyPowerDemand_TEST.ts";

        // load data files
        TimeSeriesInstances train = loadDataTS(trainDataTS);
        TimeSeriesInstances test = loadDataTS(testDataTS);

        // build classifier
        TSF tsf = new TSF(0);
        tsf.buildClassifier(train);

        // example usage to test if it's working
        double a = ClassifierTools.accuracy(test, tsf);
        System.out.println("TimeSeriesInstances Test Accuracy = " + a);
    }

    /**
     * Load data with .arff data files and convert to TSInstances to build TSF classifier
     */
    private static void tsfWithARFF() throws Exception {
        // data locations, change this to point to your datasets
        String trainDataARFF = "D:\\Documents\\Project stuff\\Datasets\\ItalyPowerDemand\\ItalyPowerDemand_TRAIN.arff";
        String testDataARFF = "D:\\Documents\\Project stuff\\Datasets\\ItalyPowerDemand\\ItalyPowerDemand_TEST.arff";

        // load data files
        Instances trainARFF = loadDataARFF(trainDataARFF);
        Instances testARFF = loadDataARFF(testDataARFF);

        // set class index
        trainARFF.setClassIndex(trainARFF.numAttributes() - 1);
        testARFF.setClassIndex(testARFF.numAttributes() - 1);

        // convert to TSInstances
        TimeSeriesInstances trainTSI = Converter.fromArff(trainARFF);
        TimeSeriesInstances testTSI = Converter.fromArff(testARFF);

        // build classifier
        TSF tsf = new TSF(0);
        tsf.buildClassifier(trainTSI);

        // example usage to test if it's working
        double a = ClassifierTools.accuracy(testTSI, tsf);
        System.out.println("Instances Test Accuracy = " + a);
    }

    /**
     * Example helper function to load in .ts data files.
     *
     * @param dataLocation string location to the .ts data file
     */
    private static TimeSeriesInstances loadDataTS(String dataLocation) {
        TimeSeriesInstances data;

        try {
            File file = new File(dataLocation);
            Reader reader = new FileReader(file); // read data from file
            TSReader tsReader = new TSReader(reader); // read ts data from reader
            data = tsReader.GetInstances(); // get TSInstances
            reader.close();
            return data;
        }
        catch (Exception e) {
            System.out.println("Exception caught: " + e);
        }

        return null;
    }

    /**
     * Example helper function to load in .arff data files.
     *
     * @param dataLocation string location to the .arff data file
     */
    private static Instances loadDataARFF(String dataLocation) {
        Instances data;

        try {
            FileReader reader = new FileReader(dataLocation);
            data = new Instances(reader);
            return data;
        }
        catch (Exception e) {
            System.out.println("Exception caught: " + e);
        }

        return null;
    }
}
