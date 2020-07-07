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


public class ShapeDTW1NN extends EnhancedAbstractClassifier {

    // hyper-parameters
    private int subsequenceLength;
    // if shapeDescriptors has length>1, then its a 'compound' shape descriptor.
    // if null, then its the 'raw' shape descriptor.
    // Supported Transformers are the following:
    // null - raw
    // PAA
    // DWT
    // Derivative
    // Slope
    // HOG1D
    // 2 of these transformers
    private Transformer [] shapeDescriptors;
    // Transformer for extracting the neighbourhoods
    private SubsequenceTransformer subsequenceTransformer;
    // NN_DTW_D for performing classification on the training data
    private NN_DTW_D nnDtwD;
    // internal attributes
    private int numClasses;

    /**
     * Private constructor so that ShapeDTW can be used out-of-the-box
     */
    public ShapeDTW1NN() {
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        this.subsequenceLength = 30;
        this.shapeDescriptors = null;
        this.subsequenceTransformer = new SubsequenceTransformer(subsequenceLength);
        this.nnDtwD = new NN_DTW_D();
    }

    public ShapeDTW1NN(int subsequenceLength,Transformer [] shapeDescriptors) {
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        this.subsequenceLength = subsequenceLength;
        this.shapeDescriptors = shapeDescriptors;
        this.subsequenceTransformer = new SubsequenceTransformer(subsequenceLength);
        this.nnDtwD = new NN_DTW_D();
    }

    public int getSubsequenceLength() {
        return subsequenceLength;
    }

    public Transformer [] getShapeDescriptors() {
        return shapeDescriptors;
    }

    public void setSubsequenceLength(int subsequenceLength) {
        this.subsequenceLength = subsequenceLength;
    }

    public void setShapeDescriptors(Transformer [] shapeDescriptors) {
        this.shapeDescriptors = shapeDescriptors;
    }

    private Instances preprocessData(Instances data) {
        Instances transformedData = this.subsequenceTransformer.transform(data);
        if (this.shapeDescriptors == null) {
            return transformedData;
        }
        for(int i=0;i<this.shapeDescriptors.length;i++) {
            transformedData = this.shapeDescriptors[i].transform(transformedData);
        }
        return transformedData;
    }

    private Instance preprocessData(Instance data) {
        Instance transformedData = this.subsequenceTransformer.transform(data);
        if (this.shapeDescriptors == null) {
            return transformedData;
        }
        for(int i=0;i<this.shapeDescriptors.length;i++) {
            transformedData = this.shapeDescriptors[i].transform(transformedData);
        }
        return transformedData;
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
