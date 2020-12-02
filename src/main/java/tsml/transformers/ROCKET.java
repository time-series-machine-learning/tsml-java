package tsml.transformers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.*;

import org.apache.commons.lang3.ArrayUtils;

import tsml.classifiers.MultiThreadable;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.TimeSeriesSummaryStatistics;
import utilities.generic_storage.Pair;
import weka.core.*;

import static utilities.ClusteringUtilities.zNormalise;
import static utilities.StatisticalUtilities.dot;
import static utilities.Utilities.extractTimeSeries;
import static utilities.multivariate_tools.MultivariateInstanceTools.*;

/**
 * ROCKET transformer. Returns the max and proportion of positive values (PPV) of n randomly initialised
 * convolutional kernels.
 *
 * @article{dempster2020rocket,
 *   title={ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels},
 *   author={Dempster, Angus and Petitjean, Fran{\c{c}}ois and Webb, Geoffrey I},
 *   journal={Data Mining and Knowledge Discovery},
 *   volume={34},
 *   number={5},
 *   pages={1454--1495},
 *   year={2020},
 *   publisher={Springer}
 * }
 *
 * Transform based on sktime python implementation by the author:
 * https://github.com/alan-turing-institute/sktime/blob/master/sktime/transformers/series_as_features/rocket.py
 *
 * @author Aaron Bostrom, Matthew Middlehurst
 */
public class ROCKET implements TrainableTransformer, Randomizable, MultiThreadable {

    private int numKernels = 10000;
    private boolean normalise = true;

    private int seed;

    private boolean multithreading = false;
    private ExecutorService ex;

    private boolean fit = false;
    private int[] candidateLengths = { 7, 9, 11 };
    private int[] numSampledDimensions, dimensions;
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

    public int getNumKernels() { return numKernels; }

    @Override
    public void setSeed(int seed) {
        this.seed = seed;
    }

    public void setNumKernels(int numKernels){ this.numKernels = numKernels; }

    public void setNormalise(boolean normalise){
        this.normalise = normalise;
    }

    @Override
    public boolean isFit() {
        return fit;
    }

    @Override
    public void enableMultiThreading(int numThreads){
        multithreading = true;
        ex = Executors.newFixedThreadPool(numThreads);
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
        double[][] output = new double[1][];
        if (multithreading){
            output[0] = transformRocketMultithread(inst.toValueArray());
        }
        else {
            output[0] = transformRocket(inst.toValueArray());
        }

        return new TimeSeriesInstance(output, inst.getLabelIndex(), inst.getClassLabels());
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

        double[] transform;
        if (multithreading){
            transform = transformRocketMultithread(data);
        }
        else{
            transform = transformRocket(data);
        }

        double[] output = new double[numKernels * 2 + 1];
        System.arraycopy(transform, 0, output, 0, numKernels * 2);
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

        int a1 = 0, a2 = 0, a3 = 0, b1, b2; // for weights, channel indices and features
        for (int i = 0; i < numKernels; i++) {
            b1 = a1 + numSampledDimensions[i] * lengths[i];
            b2 = a2 + numSampledDimensions[i];

            Pair<Double, Double> out = applyKernel(inst, ArrayUtils.subarray(weights, a1, b1), lengths[i],
                    biases[i], dilations[i], paddings[i], numSampledDimensions[i],
                    ArrayUtils.subarray(dimensions, a2, b2));
            output[a3] = out.var1;
            output[a3 + 1] = out.var2;

            a1 = b1;
            a2 = b2;
            a3 += 2;
        }

        return output;
    }

    private double[] transformRocketMultithread(double[][] inst){
        if (normalise){
            for (double[] dim : inst) {
                zNormalise(dim);
            }
        }

        ArrayList<Future<Pair<Double, Double>>> futures = new ArrayList<>(numKernels);

        int a1 = 0, a2 = 0, b1, b2;
        for (int i = 0; i < numKernels; ++i) {
            b1 = a1 + numSampledDimensions[i] * lengths[i];
            b2 = a2 + numSampledDimensions[i];

            Kernel k = new Kernel(ArrayUtils.subarray(weights, a1, b1), lengths[i], biases[i], dilations[i],
                    paddings[i], numSampledDimensions[i], ArrayUtils.subarray(dimensions, a2, b2));

            futures.add(ex.submit(new TransformThread(i, inst, k)));

            a1 = b1;
            a2 = b2;
        }

        double[] output = new double[numKernels * 2];
        int a3 = 0;
        for (Future<Pair<Double, Double>> f : futures) {
            Pair<Double, Double> out;
            try {
                out = f.get();
            } catch (Exception e) {
                e.printStackTrace();
                return null;
            }

            output[a3] = out.var1;
            output[a3 + 1] = out.var2;
            a3 += 2;
        }

        return output;
    }

    @Override
    public void fit(TimeSeriesInstances data) {
        if (multithreading){
            fitRocketMultithread(data.getMaxLength(), data.getMaxNumChannels());
        }
        else {
            fitRocket(data.getMaxLength(), data.getMaxNumChannels());
        }
    }

    @Override
    public void fit(Instances data) {
        int inputLength, numDimensions;
        if (data.checkForAttributeType(Attribute.RELATIONAL)){
            inputLength = channelLength(data);
            numDimensions = numDimensions(data);
        }
        else{
            inputLength = data.classIndex() > -1 ? data.numAttributes() - 1 : data.numAttributes();
            numDimensions = 1;
        }

        if (multithreading){
            fitRocketMultithread(inputLength, numDimensions);
        }
        else{
            fitRocket(inputLength, numDimensions);
        }
    }

    private void fitRocket(int inputLength, int numDimensions){
        Random random = new Random(seed);
        // generate random kernel lengths between 7,9 or 11, for numKernels.
        lengths = sampleLengths(random, candidateLengths, numKernels);

        // randomly select number of dimensions for each kernel
        numSampledDimensions = new int[numKernels];
        if (numDimensions == 1){
            Arrays.fill(numSampledDimensions, 1);
        }
        else{
            for (int i = 0; i < numKernels; i++) {
                int limit = Math.min(numDimensions, lengths[i]);
                // convert to base 2 log. log2(b) = log10(b) / log10(2)
                double log2 = Math.log(limit + 1) / Math.log(2.0);
                numSampledDimensions[i] = (int) Math.floor(Math.pow(2.0, uniform(random, 0, log2)));
            }
        }

        dimensions = new int[Arrays.stream(numSampledDimensions).sum()];

        // generate init values
        // weights - this should be the size of all the lengths for each dimension summed
        weights = new double[dot(lengths, numSampledDimensions)];
        biases = new double[numKernels];
        dilations = new int[numKernels];
        paddings = new int[numKernels];

        int a1 = 0, a2 = 0, b1, b2; // for weights and channel indices
        for (int i = 0; i < numKernels; i++) {
            // select weights for each dimension
            double[][] _weights = new double[numSampledDimensions[i]][];
            for (int n = 0; n < numSampledDimensions[i]; n++) {
                _weights[n] = normalDist(random, lengths[i]);
            }

            for (int n = 0; n < numSampledDimensions[i]; n++) {
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

                b2 = a2 + numSampledDimensions[i];
                for (int j = a2; j < b2; j++) {
                    dimensions[j] = al.remove(random.nextInt(al.size()));
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

    private void fitRocketMultithread(int inputLength, int numDimensions) {
        ArrayList<Future<Kernel>> futures = new ArrayList<>(numKernels);

        lengths = new int[numKernels];
        numSampledDimensions = new int[numKernels];
        int[][] tempDimensions = new int[numKernels][];
        double[][] tempWeights = new double[numKernels][];
        biases = new double[numKernels];
        dilations = new int[numKernels];
        paddings = new int[numKernels];

        for (int i = 0; i < numKernels; ++i) {
            futures.add(ex.submit(new FitThread(i, inputLength, numDimensions)));
        }

        int idx = 0;
        for (Future<Kernel> f : futures) {
            Kernel k;
            try {
                k = f.get();
            } catch (Exception e) {
                e.printStackTrace();
                return;
            }

            lengths[idx] = k.length;
            numSampledDimensions[idx] = k.numSampledDimensions;
            tempDimensions[idx] = k.dimensions;
            tempWeights[idx] = k.weights;
            biases[idx] = k.bias;
            dilations[idx] = k.dilation;
            paddings[idx] = k.padding;

            idx++;
        }

        dimensions = new int[Arrays.stream(numSampledDimensions).sum()];
        weights = new double[dot(lengths, numSampledDimensions)];

        int a1 = 0, a2 = 0; // for weights and channel indices
        for (int i = 0; i < numKernels; ++i) {
            System.arraycopy(tempWeights[i], 0, weights, a1, tempWeights[i].length);
            a1 += tempWeights[i].length;

            System.arraycopy(tempDimensions[i], 0, dimensions, a2, numSampledDimensions[i]);
            a2 += numSampledDimensions[i];
        }
    }

    private static Pair<Double, Double> applyKernel(double[][] inst, double[] weights, int length, double bias,
                                            int dilation, int padding, int numSampledDimensions, int[] dimensions) {
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
                    for (int n = 0; n < numSampledDimensions; n++) {
                        _sum = _sum + weights[j + n * numSampledDimensions] * inst[dimensions[n]][index];
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
    public void addKernels(ROCKET rocket){
        if (!fit){
            lengths = new int[0];
            numSampledDimensions = new int[0];
            dimensions = new int[0];
            weights = new double[0];
            biases = new double[0];
            dilations = new int[0];
            paddings = new int[0];
            fit = true;
        }

        lengths = ArrayUtils.addAll(lengths, rocket.lengths);
        numSampledDimensions = ArrayUtils.addAll(numSampledDimensions, rocket.numSampledDimensions);
        dimensions = ArrayUtils.addAll(dimensions, rocket.dimensions);
        weights = ArrayUtils.addAll(weights, rocket.weights);
        biases = ArrayUtils.addAll(biases, rocket.biases);
        dilations = ArrayUtils.addAll(dilations, rocket.dilations);
        paddings = ArrayUtils.addAll(paddings, rocket.paddings);

        numKernels += rocket.numKernels;
    }

    private static class Kernel {
        int numSampledDimensions;
        int[] dimensions;
        int length, dilation, padding;
        double[] weights;
        double bias;

        public Kernel() { }

        public Kernel(double[] weights, int length, double bias, int dilation, int padding, int numSampledDimensions,
                      int[] dimensions) {
            this.weights = weights;
            this.length = length;
            this.bias = bias;
            this.dilation = dilation;
            this.padding = padding;
            this.numSampledDimensions = numSampledDimensions;
            this.dimensions = dimensions;
        }
    }

    private class TransformThread implements Callable<Pair<Double, Double>> {
        int i;
        double[][] inst;
        Kernel k;

        public TransformThread(int i, double[][] inst, Kernel k){
            this.i = i;
            this.inst = inst;
            this.k = k;
        }

        @Override
        public Pair<Double, Double> call() {
            int inputLength = inst[0].length;
            int outputLength = (inputLength + (2 * k.padding)) - ((k.length - 1) * k.dilation);

            double _ppv = 0;
            double _max = Double.MIN_VALUE;
            int end = (inputLength + k.padding) - ((k.length - 1) * k.dilation);

            for (int i = -k.padding; i < end; i++) {
                double _sum = k.bias;
                int index = i;

                for (int j = 0; j < k.length; j++) {
                    if (index > -1 && index < inputLength) {
                        for (int n = 0; n < k.numSampledDimensions; n++) {
                            _sum = _sum + k.weights[j + n * k.numSampledDimensions] * inst[k.dimensions[n]][index];
                        }
                    }
                    index = index + k.dilation;
                }

                if (_sum > _max)
                    _max = _sum;

                if (_sum > 0)
                    _ppv += 1;
            }

            return new Pair<>(_ppv / outputLength, _max);
        }
    }

    private class FitThread implements Callable<Kernel>{
        int i;
        int inputLength;
        int numDimensions;

        public FitThread(int i, int inputLength, int numDimensions){
            this.i = i;
            this.inputLength = inputLength;
            this.numDimensions = numDimensions;
        }

        @Override
        public Kernel call() {
            Kernel k = new Kernel();
            Random random = new Random(seed + i * numKernels);

            k.length = candidateLengths[random.nextInt(candidateLengths.length)];

            // randomly select number of dimensions for each kernel
            if (numDimensions == 1){
                k.numSampledDimensions = 1;
            }
            else{
                int limit = Math.min(numDimensions, k.length);
                // convert to base 2 log. log2(b) = log10(b) / log10(2)
                double log2 = Math.log(limit + 1) / Math.log(2.0);
                k.numSampledDimensions = (int) Math.floor(Math.pow(2.0, uniform(random, 0, log2)));
            }

            // select weights for each dimension
            double[][] _weights = new double[k.numSampledDimensions][];
            for (int n = 0; n < k.numSampledDimensions; n++) {
                _weights[n] = normalDist(random, k.length);
            }

            k.weights = new double[k.length * k.numSampledDimensions];
            int a1 = 0, b1;
            for (int n = 0; n < k.numSampledDimensions; n++) {
                b1 = a1 + k.length;
                double mean = TimeSeriesSummaryStatistics.mean(_weights[n]);
                for (int j = a1; j < b1; ++j) {
                    k.weights[j] = _weights[n][j - a1] - mean;
                }
                a1 = b1;
            }

            // randomly select dimensions for kernel
            k.dimensions = new int[k.numSampledDimensions];
            if (numDimensions > 1){
                ArrayList<Integer> al = new ArrayList<>(numDimensions);
                for (int n = 0; n < numDimensions; n++) {
                    al.add(n);
                }

                for (int j = 0; j < k.numSampledDimensions; j++) {
                    k.dimensions[j] = al.remove(random.nextInt(al.size()));
                }
            }

            // draw uniform random sample from 0-1 and shift it to -1 to 1.
            k.bias = (random.nextDouble() * 2.0) - 1.0;

            double value = (double) (inputLength - 1) / (double) (k.length - 1);
            // convert to base 2 log. log2(b) = log10(b) / log10(2)
            double log2 = Math.log(value) / Math.log(2.0);
            k.dilation = (int) Math.floor(Math.pow(2.0, uniform(random, 0, log2)));

            k.padding = random.nextInt(2) == 1 ? Math.floorDiv((k.length - 1) * k.dilation, 2) : 0;

            return k;
        }
    }
}
