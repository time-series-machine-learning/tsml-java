package tsml.transformers;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import org.apache.commons.lang3.ArrayUtils;

import scala.collection.parallel.ParIterableLike.Collect;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.TimeSeriesSummaryStatistics;
import utilities.generic_storage.Pair;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;

public class ROCKET implements TrainableTransformer, Randomizable {

    int seed;

    int numKernels;
    int[] candidateLengths = { 7, 9, 11 };
    int[] lengths, dilations, paddings;
    double[] weights, biases;

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
        // apply kernels to the dataset.
        double[][] output = new double[1][numKernels * 2]; // 2 features per kernel

        int a1 = 0; // for weights
        int a2 = 0; // for features

        for (int j = 0; j < numKernels; j++) {

            int b1 = a1 + lengths[j];
            int b2 = a2 + 2;

            //TODO: look at multivariate ROCKET 
            //univariate dim 0.
            Pair<Double, Double> out = applyKernel(inst.get(0), ArrayUtils.subarray(weights, a1, b1),
                    lengths[j], biases[j], dilations[j], paddings[j]);

            output[0][a2] = out.var1;
            output[0][a2 + 1] = out.var2;

            a1 = b1;
            a2 = b2;
        }
        
        return new TimeSeriesInstance(output, inst.getLabelIndex());
    }

    @Override
    public boolean isFit() {
        // TODO Auto-generated method stub
        return false;
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
            double[] weights = this.normalDist(random, lengths[i]);
            double mean = TimeSeriesSummaryStatistics.mean(weights);

            b1 = a1 + lengths[i];
            for (int j = a1; j < b1; ++j) {
                weights[j] = weights[j] - mean;
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

}
