package tsml.examples.ts_examples;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.ts_fileIO.TSReader;
import tsml.data_containers.ts_fileIO.TSWriter;
import tsml.data_containers.utilities.TimeSeriesResampler;
import tsml.data_containers.utilities.TimeSeriesResampler.TrainTest;

public class FileWritingExample {

    public static void example1() throws FileNotFoundException, IOException {
        String m_local_path = "D:\\Work\\Data\\Multivariate_ts\\";
        String m_local_path_orig = "D:\\Work\\Data\\Multivariate_arff\\";
        
        String dataset = "BasicMotions";
        String filepath = m_local_path + dataset + "\\" + dataset;
        String filepath_Arff = m_local_path_orig + dataset + "\\" + dataset;

        TSReader ts_reader = new TSReader(new FileReader(new File(filepath + "_TRAIN" + ".ts")));
        TimeSeriesInstances ts_train_data = ts_reader.GetInstances();

        ts_reader = new TSReader(new FileReader(new File(filepath + "_TEST" + ".ts")));
        TimeSeriesInstances ts_test_data = ts_reader.GetInstances();

        TrainTest out2 = TimeSeriesResampler.resampleTrainTest(ts_train_data, ts_test_data, 1);


        TSWriter writer = new TSWriter(new File(dataset + "_TRAIN " + 1 + ".ts"));
        //TSWriter writer = new TSWriter();
        //writer.setDestination(System.out);
        writer.setData(out2.train);
        writer.writeBatch();

        writer = new TSWriter(new File(dataset + "_TEST " + 1 + ".ts"));
        //writer = new TSWriter();
        //writer.setDestination(System.out);
        writer.setData(out2.test);
        writer.writeBatch();
    }

    public static void main(String[] args) throws FileNotFoundException, IOException {
        example1();
    }
}