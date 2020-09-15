package tsml.transformers;

import java.util.Arrays;
import java.util.Random;

import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.TimeSeriesSummaryStatistics;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;

public class ROCKET implements TrainableTransformer, Randomizable {

    int seed;

    int numKernels;
    int[] candidateLengths = {7,9,11};
    int[] lengths;
    double[] weights, biases, dilations, paddings;

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
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public boolean isFit() {
        // TODO Auto-generated method stub
        return false;
    }


    @Override
    public void fit(TimeSeriesInstances data) {

        int inputlength = data.getMinLength();

        // TODO Auto-generated method stub

        Random random = new Random(this.seed);
        //generate random kernel lengths between 7,9 or 11, for numKernels.
        //lengths =  random.choice(candidateLengths, numKernels);

        
        //generate init values
        //weights - this should be the size of all the lengths summed
        weights= new double[Arrays.stream(lengths).sum()];
        biases = new double[numKernels];
        dilations = new double[numKernels];
        paddings = new double[numKernels];

        int a1 =0;
        int b1 = 0;
        for(int i=0; i < numKernels; i++){
            double[] weights = this.normalDist(random, lengths[i]);
            double mean = TimeSeriesSummaryStatistics.mean(weights);

            b1 = a1 + lengths[i]; 
            for(int j=a1; j<b1; ++j){
                weights[j] = weights[j] - mean;
            }
            
            //draw uniform random sample from 0-1 and shift it to -1 to 1.
            biases[i] = (random.nextDouble() * 2.0) - 1.0; 

            double value = (double)(inputlength - 1) / (double)(lengths[i] - 1);
            //convert to base 2 log. log2(b) = log10(b) / log10(2)
            double log2 = Math.log(value) / Math.log(2.0);
            dilations[i] = Math.floor(Math.pow(2.0, uniform(random, 0, log2)));

            paddings[i] = random.nextInt(2) == 1 ? Math.floorDiv((lengths[i] - 1) * (int)dilations[i], 2) : 0;



            a1 = b1;
        }

    }


    //TODO: look up better Affine methods - not perfect but will do
    double uniform(Random rand, double a, double b){
        return a + rand.nextDouble() * (b - a + 1);
    }

    double[] normalDist(Random rand, int size){
        double[] out = new double[size];
        for(int i=0; i<size; ++i)
            out[i] = rand.nextGaussian();
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
