package tsml.examples.ts_examples;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import experiments.data.DatasetLoading;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.ts_fileIO.TSReader;
import tsml.data_containers.utilities.TimeSeriesResampler;
import tsml.data_containers.utilities.TimeSeriesResampler.TrainTest;
import utilities.InstanceTools;
import utilities.multivariate_tools.MultivariateInstanceTools;
import weka.core.Instances;

public class ResamplingExamples {

    public static void example1() throws FileNotFoundException, IOException {
        String local_path = "D:\\Work\\Data\\Univariate_ts\\";
        String local_path_orig = "D:\\Work\\Data\\Univariate_arff\\";
        
        String dataset = "ItalyPowerDemand";
        String filepath = local_path + dataset + "\\" + dataset;
        String filepath_Arff = local_path_orig + dataset + "\\" + dataset;

        TSReader ts_reader = new TSReader(new FileReader(new File(filepath + "_TRAIN" + ".ts")));
        TimeSeriesInstances ts_train_data = ts_reader.GetInstances();

        ts_reader = new TSReader(new FileReader(new File(filepath + "_TEST" + ".ts")));
        TimeSeriesInstances ts_test_data = ts_reader.GetInstances();

        Instances train_data = DatasetLoading.loadData(filepath_Arff + "_TRAIN" + ".arff");
        Instances test_data = DatasetLoading.loadData(filepath_Arff + "_TEST" + ".arff");

        Instances[] out1 = InstanceTools.resampleTrainAndTestInstances(train_data, test_data, 1);
        System.out.println(out1[0].instance(0));
        System.out.println(out1[1].instance(0));

        TrainTest out2 = TimeSeriesResampler.resampleTrainTest(ts_train_data, ts_test_data, 1);
        System.out.println(out2.train.get(0));
        System.out.println(out2.test.get(0));
    }

    public static void example2() throws FileNotFoundException, IOException {
        String m_local_path = "D:\\Work\\Data\\Multivariate_ts\\";
        String m_local_path_orig = "D:\\Work\\Data\\Multivariate_arff\\";
        
        String dataset = "BasicMotions";
        String filepath = m_local_path + dataset + "\\" + dataset;
        String filepath_Arff = m_local_path_orig + dataset + "\\" + dataset;

        TSReader ts_reader = new TSReader(new FileReader(new File(filepath + "_TRAIN" + ".ts")));
        TimeSeriesInstances ts_train_data = ts_reader.GetInstances();

        ts_reader = new TSReader(new FileReader(new File(filepath + "_TEST" + ".ts")));
        TimeSeriesInstances ts_test_data = ts_reader.GetInstances();

        Instances train_data = DatasetLoading.loadData(filepath_Arff + "_TRAIN" + ".arff");
        Instances test_data = DatasetLoading.loadData(filepath_Arff + "_TEST" + ".arff");

        Instances[] out1 = MultivariateInstanceTools.resampleMultivariateTrainAndTestInstances_old(train_data, test_data, 1);
        System.out.println(out1[0].instance(0));
        System.out.println(out1[1].instance(0));

        TrainTest out2 = TimeSeriesResampler.resampleTrainTest(ts_train_data, ts_test_data, 1);
        System.out.println(out2.train.get(0));
        System.out.println(out2.test.get(0));
    }

    public static void main(String[] args) throws Exception {
        example1();

        System.out.println("----------------------------------------------------");

        example2();
    }
    

}