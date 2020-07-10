package tsml.contrib;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.contrib.transformers.SubsequenceTransformer;
import tsml.transformers.Derivative;
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
     * well as the shape descriptor function (if not null).
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
        return this.shapeDescriptor.transform(transformedData);
    }

    /**
     * Private method for performing the subsequence extraction on an instance as
     * well as the shape descriptor function (if not null).
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
        return this.shapeDescriptor.transform(transformedData);
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
}
