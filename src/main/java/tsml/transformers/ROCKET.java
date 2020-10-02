package tsml.transformers;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.lang3.ArrayUtils;

import core.contracts.Dataset;
import experiments.data.DatasetLoading;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.ts_fileIO.TSReader;
import tsml.data_containers.utilities.Converter;
import tsml.data_containers.utilities.TimeSeriesSummaryStatistics;
import utilities.generic_storage.Pair;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;

public class ROCKET implements TrainableTransformer, Randomizable {

    int seed;
    boolean fit = false;
    int numKernels = 100; //either 100. 1000. or 10,000
    int[] candidateLengths = { 7, 9, 11 };
    int[] lengths, dilations, paddings;
    double[] weights, biases;

    public ROCKET(int numKernels){
        this.numKernels = numKernels;
    }

    @Override
    public Instance transform(Instance inst) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Instances determineOutputFormat(Instances data) throws IllegalArgumentException {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void fit(Instances data) {
        // TODO Auto-generated method stub

    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        int dims = inst.getNumDimensions();

        // apply kernels to the dataset.
        double[][] output = new double[dims][numKernels * 2]; // 2 features per kernel

        int a1 = 0; // for weights
        int a2 = 0; // for features

        for (int j = 0; j < numKernels; j++) {
            int b1 = a1 + lengths[j];
            int b2 = a2 + 2;

            for (int i = 0; i < dims; i++) {
                Pair<Double, Double> out = applyKernel(inst.get(i), ArrayUtils.subarray(weights, a1, b1), lengths[j],
                        biases[j], dilations[j], paddings[j]);
                output[i][a2] = out.var1;
                output[i][a2 + 1] = out.var2;
            }

            a1 = b1;
            a2 = b2;
        }

        return new TimeSeriesInstance(output, inst.getLabelIndex());
    }

    @Override
    public boolean isFit() {
        return fit;
    }

    @Override
    public void fit(TimeSeriesInstances data) {

        int inputlength = data.getMinLength();

        Random random = new Random(this.seed);
        // generate random kernel lengths between 7,9 or 11, for numKernels.
        lengths = sampleLengths(random, candidateLengths, numKernels);
        // generate init values
        // weights - this should be the size of all the lengths summed
        weights = new double[Arrays.stream(lengths).sum()];
        biases = new double[numKernels];
        dilations = new int[numKernels];
        paddings = new int[numKernels];

        int a1 = 0;
        int b1 = 0;
        for (int i = 0; i < numKernels; i++) {
            double[] _weights = this.normalDist(random, lengths[i]);
            double mean = TimeSeriesSummaryStatistics.mean(_weights);

            b1 = a1 + lengths[i];
            for (int j = a1; j < b1; ++j) {
                weights[j] = _weights[j-a1] - mean;
            }

            // draw uniform random sample from 0-1 and shift it to -1 to 1.
            biases[i] = (random.nextDouble() * 2.0) - 1.0;

            double value = (double) (inputlength - 1) / (double) (lengths[i] - 1);
            // convert to base 2 log. log2(b) = log10(b) / log10(2)
            double log2 = Math.log(value) / Math.log(2.0);
            dilations[i] = (int) Math.floor(Math.pow(2.0, uniform(random, 0, log2)));

            paddings[i] = random.nextInt(2) == 1 ? Math.floorDiv((lengths[i] - 1) * (int) dilations[i], 2) : 0;

            a1 = b1;
        }
        
        fit = true;
    }

    public Pair<Double, Double> applyKernel(TimeSeries inst, double[] weights, int length, double bias, int dilation,
            int padding) {

        int inputLength = inst.getSeriesLength();
        int outputLength = (inputLength + (2 * padding)) - ((length - 1) * dilation);

        double _ppv = 0;
        double _max = Double.MIN_VALUE;
        int end = (inputLength + padding) - ((length - 1) * dilation);

        for (int i = -padding; i < end; i++) {
            double _sum = bias;
            int index = i;

            for (int j = 0; j < length; j++) {
                if (index > -1 && index < inputLength)
                    _sum = _sum + weights[j] * inst.get(index);
                index = index + dilation;
            }

            if (_sum > _max)
                _max = _sum;

            if (_sum > 0)
                _ppv += 1;
        }

        return new Pair<>(_ppv / outputLength, _max);
    }

    // TODO: look up better Affine methods - not perfect but will do
    double uniform(Random rand, double a, double b) {
        return a + rand.nextDouble() * (b - a + 1);
    }

    double[] normalDist(Random rand, int size) {
        double[] out = new double[size];
        for (int i = 0; i < size; ++i)
            out[i] = rand.nextGaussian();
        return out;
    }

    int[] sampleLengths(Random random, int[] samples, int size) {
        int[] out = new int[size];
        for (int i = 0; i < size; ++i) {
            out[i] = samples[random.nextInt(samples.length)];
        }
        return out;
    }

    @Override
    public void setSeed(int seed) {
        this.seed = seed;
    }

    @Override
    public int getSeed() {
        return seed;
    }

    public static void main(String[] args) throws Exception {
        
        // Aarons local path for testing.
        String local_path = "D:\\Work\\Data\\Univariate_ts\\"; // Aarons local path for testing.
        // String m_local_path = "D:\\Work\\Data\\Multivariate_ts\\";
        // String m_local_path_orig = "D:\\Work\\Data\\Multivariate_arff\\";
        String dataset_name = "Chinatown";

        TSReader ts_reader = new TSReader(new FileReader(new File(local_path + dataset_name + File.separator + dataset_name + "_TRAIN.ts")));
        TimeSeriesInstances train = ts_reader.GetInstances();

        ts_reader = new TSReader(new FileReader(new File(local_path + dataset_name + File.separator + dataset_name + "_TEST.ts")));
        TimeSeriesInstances test = ts_reader.GetInstances();

        ROCKET rocket = new ROCKET(10000);
        TimeSeriesInstances ttrain = rocket.fitTransform(train);
        TimeSeriesInstances ttest = rocket.transform(test);

        Logistic clf = new Logistic();
        clf.buildClassifier(Converter.toArff(ttrain));
        double acc = utilities.ClassifierTools.accuracy(Converter.toArff(ttest), clf);
        System.out.println("acc: " + acc);


    }
}
