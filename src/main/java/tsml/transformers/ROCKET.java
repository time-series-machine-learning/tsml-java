package tsml.transformers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.lang3.ArrayUtils;

import experiments.data.DatasetLoading;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import tsml.data_containers.utilities.TimeSeriesSummaryStatistics;
import utilities.generic_storage.Pair;
import weka.classifiers.functions.Logistic;
import weka.core.*;

import static utilities.ClusteringUtilities.zNormalise;
import static utilities.InstanceTools.resampleTrainAndTestInstances;
import static utilities.StatisticalUtilities.dot;
import static utilities.StatisticalUtilities.mean;
import static utilities.Utilities.extractTimeSeries;
import static utilities.multivariate_tools.MultivariateInstanceTools.*;

/**
 *
 *
 * @author Aaron Bostrom, Matthew Middlehurst
 */
public class ROCKET implements TrainableTransformer, Randomizable {

    private int numKernels = 10000;
    private boolean normalise = true;

    private int seed;

    private boolean fit = false;
    private int[] candidateLengths = { 7, 9, 11 };
    private int[] numDimensionIndices, dimensionIndicies;
    private int[] lengths, dilations, paddings;
    private double[] weights, biases;

    public ROCKET(){ }

    public ROCKET(int numKernels){
        this.numKernels = numKernels;
    }

    @Override
    public int getSeed() {
        return seed;
    }

    @Override
    public void setSeed(int seed) {
        this.seed = seed;
    }

    public void setNormalise(boolean normalise){
        this.normalise = normalise;
    }

    @Override
    public boolean isFit() {
        return fit;
    }

    @Override
    public Instances determineOutputFormat(Instances data) throws IllegalArgumentException {
        ArrayList<Attribute> atts = new ArrayList<>();
        for (int i = 0; i < numKernels * 2; i++) {
            atts.add(new Attribute("att" + i));
        }
        if (data.classIndex() >= 0) atts.add(data.classAttribute());
        Instances transformedData = new Instances("ROCKETTransform", atts, data.numInstances());
        if (data.classIndex() >= 0) transformedData.setClassIndex(transformedData.numAttributes() - 1);
        return transformedData;
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        double[][] output = new double[1][]; // 2 features per kernel
        output[0] = transformRocket(inst.toValueArray());

        return new TimeSeriesInstance(output, inst.getLabelIndex());
    }

    @Override
    public Instance transform(Instance inst) {
        double[][] data;
        if (inst.dataset().checkForAttributeType(Attribute.RELATIONAL)) {
            data = convertMultiInstanceToArrays(splitMultivariateInstance(inst));
        }
        else{
            data = new double[1][];
            data[0] = extractTimeSeries(inst);
        }

        double[] output = new double[numKernels * 2 + 1];
        System.arraycopy(transformRocket(data), 0, output, 0, numKernels * 2);
        output[output.length - 1] = inst.classValue();

        return new DenseInstance(1, output);
    }

    private double[] transformRocket(double[][] inst){
        if (normalise){
            for (double[] dim : inst) {
                zNormalise(dim);
            }
        }

        // apply kernels to the dataset.
        double[] output = new double[numKernels * 2]; // 2 features per kernel

        int a1 = 0, a2 = 0, a3 = 0; // for weights, channel indices and features
        int b1, b2, b3;

        for (int j = 0; j < numKernels; j++) {
            b1 = a1 + numDimensionIndices[j] * lengths[j];
            b2 = a2 + numDimensionIndices[j];
            b3 = a3 + 2;

            Pair<Double, Double> out = applyKernel(inst, ArrayUtils.subarray(weights, a1, b1), lengths[j],
                    biases[j], dilations[j], paddings[j], numDimensionIndices[j],
                    ArrayUtils.subarray(dimensionIndicies, a2, b2));
            output[a3] = out.var1;
            output[a3 + 1] = out.var2;

            a1 = b1;
            a2 = b2;
            a3 = b3;
        }

        return output;
    }

    @Override
    public void fit(TimeSeriesInstances data) {
        fitRocket(data.getMaxLength(), data.getMaxNumChannels());
    }

    @Override
    public void fit(Instances data) {
        if (data.checkForAttributeType(Attribute.RELATIONAL)){
            fitRocket(channelLength(data), numDimensions(data));
        }
        else{
            int inputLength = data.classIndex() > -1 ? data.numAttributes() - 1 : data.numAttributes();
            fitRocket(inputLength, 1);
        }
    }

    private void fitRocket(int inputLength, int numDimensions){
        Random random = new Random(seed);
        // generate random kernel lengths between 7,9 or 11, for numKernels.
        lengths = sampleLengths(random, candidateLengths, numKernels);

        // randomly select number of dimensions for each kernel
        numDimensionIndices = new int[numKernels];
        if (numDimensions == 1){
            Arrays.fill(numDimensionIndices, 1);
        }
        else{
            for (int i = 0; i < numKernels; i++) {
                int limit = Math.min(numDimensions, lengths[i]);
                // convert to base 2 log. log2(b) = log10(b) / log10(2)
                double log2 = Math.log(limit + 1) / Math.log(2.0);
                numDimensionIndices[i] = (int) Math.floor(Math.pow(2.0, uniform(random, 0, log2)));
            }
        }

        dimensionIndicies = new int[Arrays.stream(numDimensionIndices).sum()];

        // generate init values
        // weights - this should be the size of all the lengths for each dimension summed
        weights = new double[dot(lengths, numDimensionIndices)];
        biases = new double[numKernels];
        dilations = new int[numKernels];
        paddings = new int[numKernels];

        int a1 = 0, a2 = 0; // for weights and channel indices
        int b1, b2;
        for (int i = 0; i < numKernels; i++) {
            // select weights for each dimension
            double[][] _weights = new double[numDimensionIndices[i]][];
            for (int n = 0; n < numDimensionIndices[i]; n++) {
                _weights[n] = normalDist(random, lengths[i]);
            }

            for (int n = 0; n < numDimensionIndices[i]; n++) {
                b1 = a1 + lengths[i];
                double mean = TimeSeriesSummaryStatistics.mean(_weights[n]);
                for (int j = a1; j < b1; ++j) {
                    weights[j] = _weights[n][j - a1] - mean;
                }
                a1 = b1;
            }

            // randomly select dimensions for kernel
            if (numDimensions > 1){
                ArrayList<Integer> al = new ArrayList<>(numDimensions);
                for (int n = 0; n < numDimensions; n++) {
                    al.add(n);
                }

                b2 = a2 + numDimensionIndices[i];
                for (int j = a2; j < b2; j++) {
                    dimensionIndicies[j] = al.remove(random.nextInt(al.size()));
                }
                a2 = b2;
            }

            // draw uniform random sample from 0-1 and shift it to -1 to 1.
            biases[i] = (random.nextDouble() * 2.0) - 1.0;

            double value = (double) (inputLength - 1) / (double) (lengths[i] - 1);
            // convert to base 2 log. log2(b) = log10(b) / log10(2)
            double log2 = Math.log(value) / Math.log(2.0);
            dilations[i] = (int) Math.floor(Math.pow(2.0, uniform(random, 0, log2)));

            paddings[i] = random.nextInt(2) == 1 ? Math.floorDiv((lengths[i] - 1) * dilations[i], 2) : 0;
        }

        fit = true;
    }

    private static Pair<Double, Double> applyKernel(double[][] inst, double[] weights, int length, double bias,
                                            int dilation, int padding, int numDimensionIndicies,
                                            int[] dimensionIndicies) {
        int inputLength = inst[0].length;
        int outputLength = (inputLength + (2 * padding)) - ((length - 1) * dilation);

        double _ppv = 0;
        double _max = Double.MIN_VALUE;
        int end = (inputLength + padding) - ((length - 1) * dilation);

        for (int i = -padding; i < end; i++) {
            double _sum = bias;
            int index = i;

            for (int j = 0; j < length; j++) {
                if (index > -1 && index < inputLength) {
                    for (int n = 0; n < numDimensionIndicies; n++) {
                        _sum = _sum + weights[j + n * numDimensionIndicies] * inst[dimensionIndicies[n]][index];
                    }
                }
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
    private static double uniform(Random rand, double a, double b) {
        return a + rand.nextDouble() * (b - a);
    }

    private static double[] normalDist(Random rand, int size) {
        double[] out = new double[size];
        for (int i = 0; i < size; ++i)
            out[i] = rand.nextGaussian();
        return out;
    }

    private static int[] sampleLengths(Random random, int[] samples, int size) {
        int[] out = new int[size];
        for (int i = 0; i < size; ++i) {
            out[i] = samples[random.nextInt(samples.length)];
        }
        return out;
    }

    public static void main(String[] args) throws Exception {
//        String local_path = "Z:\\ArchiveData\\Univariate_ts\\";
//        String dataset_name = "ItalyPowerDemand";
//
//        TSReader ts_reader = new TSReader(new FileReader(new File(local_path + dataset_name + File.separator
//                + dataset_name + "_TRAIN.ts")));
//        TimeSeriesInstances train = ts_reader.GetInstances();
//
//        ts_reader = new TSReader(new FileReader(new File(local_path + dataset_name + File.separator
//                + dataset_name + "_TEST.ts")));
//        TimeSeriesInstances test = ts_reader.GetInstances();

        int fold = 0;

        //Minimum working example
        String dataset = "GunPoint";
        Instances train = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\"+dataset+
                "\\"+dataset+"_TRAIN.arff");
        Instances test = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\"+dataset+
                "\\"+dataset+"_TEST.arff");
        Instances[] data = resampleTrainAndTestInstances(train, test, fold);
        train = data[0];
        test = data[1];

//        String dataset2 = "ERing";
//        Instances trainMV = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Multivariate_arff\\"+dataset2+
//                "\\"+dataset2+"_TRAIN.arff");
//        Instances testMV = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Multivariate_arff\\"+dataset2+
//                "\\"+dataset2+"_TEST.arff");
//        Instances[] data2 = resampleMultivariateTrainAndTestInstances(trainMV, testMV, fold);
//        trainMV = data2[0];
//        testMV = data2[1];

//        ROCKET rocket = new ROCKET(1000);
//        rocket.seed = fold;
//        TimeSeriesInstances ttrain = rocket.fitTransform(Converter.fromArff(trainMV));
//        TimeSeriesInstances ttest = rocket.transform(Converter.fromArff(testMV));
//
//        Logistic clf = new Logistic();
//        clf.buildClassifier(Converter.toArff(ttrain));
//        double acc = utilities.ClassifierTools.accuracy(Converter.toArff(ttest), clf);
//        System.out.println("acc: " + acc);

        ROCKET rocket2 = new ROCKET(1000);
        rocket2.seed = fold;
        Instances ttrain2 = rocket2.fitTransform(train);
        Instances ttest2 = rocket2.transform(test);

        Logistic clf2 = new Logistic();
        clf2.buildClassifier(ttrain2);
        double acc2 = utilities.ClassifierTools.accuracy(ttest2, clf2);
        System.out.println("acc: " + acc2);
    }
}
