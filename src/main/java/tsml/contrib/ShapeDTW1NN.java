package tsml.contrib;

import experiments.data.DatasetLoading;
import statistics.simulators.DictionaryModel;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.contrib.transformers.DWTTransformer;
import tsml.contrib.transformers.HOG1DTransformer;
import tsml.contrib.transformers.SlopeTransformer;
import tsml.contrib.transformers.SubsequenceTransformer;
import tsml.transformers.Derivative;
import tsml.transformers.DimensionIndependentTransformer;
import tsml.transformers.PAA;
import tsml.transformers.Transformer;
import weka.core.Instance;
import weka.core.Instances;
import java.util.concurrent.TimeUnit;
import tsml.classifiers.multivariate.NN_DTW_D;

/**
 * The ShapeDTW classifier works by initially extracting a set of subsequences
 * describing local neighbourhoods around each data point in a time series.
 * These subsequences are then passed into a shape descriptor function that
 * transforms these local neighbourhoods into a new representation. This
 * new representation is then sent into DTW with 1-NN.
 *
 * @author Vincent Nicholson
 *
 */
public class ShapeDTW1NN extends EnhancedAbstractClassifier {

    // hyper-parameters
    private int subsequenceLength;
    // if shapeDescriptor is null, then its the 'raw' shape descriptor.
    // Supported Transformers are the following:
    // null - raw
    // PAA
    // DWT
    // Derivative
    // Slope
    // HOG1D
    // Compound (class that performs two transformers and concatenates their results together).
    private Transformer shapeDescriptor;
    // Transformer for extracting the neighbourhoods
    private SubsequenceTransformer subsequenceTransformer;
    // NN_DTW_D for performing classification on the training data
    private NN_DTW_D nnDtwD;
    private DimensionIndependentTransformer d;

    /**
     * Private constructor with settings:
     * subsequenceLength = 30
     * shapeDescriptorFunction = null (so 'raw' is used)
     */
    public ShapeDTW1NN() {
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        this.subsequenceLength = 30;
        this.shapeDescriptor = null;
        this.subsequenceTransformer = new SubsequenceTransformer(subsequenceLength);
        this.nnDtwD = new NN_DTW_D();
    }

    public ShapeDTW1NN(int subsequenceLength,Transformer shapeDescriptor) {
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        this.subsequenceLength = subsequenceLength;
        this.shapeDescriptor = shapeDescriptor;
        this.subsequenceTransformer = new SubsequenceTransformer(subsequenceLength);
        this.nnDtwD = new NN_DTW_D();
    }

    public int getSubsequenceLength() {
        return subsequenceLength;
    }

    public Transformer getShapeDescriptors() {
        return shapeDescriptor;
    }

    public void setSubsequenceLength(int subsequenceLength) {
        this.subsequenceLength = subsequenceLength;
    }

    public void setShapeDescriptors(Transformer shapeDescriptors) {
        this.shapeDescriptor = shapeDescriptors;
    }

    /**
     * Private method for performing the subsequence extraction on a set of instances as
     * well as the shape descriptor function for training (if not null).
     *
     * @param data
     * @return
     */
    private Instances preprocessData(Instances data) {
        Instances transformedData = this.subsequenceTransformer.transform(data);
        //If shape descriptor is null aka 'raw', use the subsequences.
        if (this.shapeDescriptor == null) {
            return transformedData;
        }
        this.d = new DimensionIndependentTransformer(this.shapeDescriptor);
        Instances res = d.transform(transformedData);
        return res;
    }

    /**
     * Private method for performing the subsequence extraction on an instance as
     * well as the shape descriptor function for testing (if not null).
     *
     * @param data
     * @return
     */
    private Instance preprocessData(Instance data) {
        Instance transformedData = this.subsequenceTransformer.transform(data);
        //If shape descriptor is null aka 'raw', use the subsequences.
        if (this.shapeDescriptor == null) {
            return transformedData;
        }
        Instance res = this.d.transform(transformedData);
        return res;
    }

    @Override
    public void buildClassifier(Instances trainInst) throws Exception {
        // Check the data
        this.getCapabilities().testWithFail(trainInst);
        // Record the build time.
        long buildTime = System.nanoTime();
        // Train the classifier
        Instances transformedData = this.preprocessData(trainInst);
        this.nnDtwD.buildClassifier(transformedData);
        // Store the timing results.
        buildTime = System.nanoTime() - buildTime ;
        this.trainResults.setBuildTime(buildTime);
        this.trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
    }

    @Override
    public double [] distributionForInstance(Instance testInst) throws Exception {
        Instance transformedData = this.preprocessData(testInst);
        return this.nnDtwD.distributionForInstance(transformedData);
    }

    @Override
    public double classifyInstance(Instance testInst) throws Exception {
        Instance transformedData = this.preprocessData(testInst);
        return this.nnDtwD.classifyInstance(transformedData);
    }

    public static void main(String[] args) throws Exception {
        Instances [] data = DatasetLoading.sampleItalyPowerDemand(0);

        PAA p = new PAA();
        DWTTransformer d = new DWTTransformer();
        Derivative de = new Derivative();
        SlopeTransformer sl = new SlopeTransformer();
        HOG1DTransformer h = new HOG1DTransformer();
        ShapeDTW1NN s = new ShapeDTW1NN(30,null);
        System.out.println(calculateAccuracy(s,data));
    }

    private static double calculateAccuracy(ShapeDTW1NN s, Instances [] data) throws Exception {
        Instances train = data[0];
        Instances test = data[1];

        s.buildClassifier(train);
        int correct = 0;
        for(int i=0;i<test.numInstances();i++) {
            double predict = s.classifyInstance(test.get(i));
            if(predict == test.get(i).classValue()) {
                correct++;
            }
        }
        return (double) correct/(double) test.numInstances();
    }
}
