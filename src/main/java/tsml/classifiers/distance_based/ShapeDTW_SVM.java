package tsml.classifiers.distance_based;

import experiments.data.DatasetLoading;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.transformers.ShapeDTWFeatures;
import tsml.transformers.Subsequences;
import utilities.samplers.RandomStratifiedSampler;
import weka.classifiers.functions.SMO;
import weka.core.Instance;
import weka.core.Instances;

import java.util.concurrent.TimeUnit;

/*
 * This class extends on the idea of ShapeDTW_1NN by transforming
 * a test instance into ShapeDTW distance features (note that it uses
 * the default shape descriptor which is 'raw'). It calculates the test
 * instance's distance from a reference set and these become the features
 * for the instance. This transformed instance is then fed into a
 * SVM classifier. The same task is done on the training set.
 */
public class ShapeDTW_SVM extends EnhancedAbstractClassifier {

    private Instances trainingSet;
    private Instances referenceSet;
    // Defines the split from training to reference, 0.5 indicates
    // 0.5 train, 0.5 reference. 0.4 indicates 0.4 train, 0.6 reference
    // and so on.
    private double trainRefSplit = 0.5;
    private Subsequences subsequenceTransformer;
    //The transformer used to produce the shape dtw features.
    private ShapeDTWFeatures sdtwFeats;
    //The stratified sampler
    private RandomStratifiedSampler sampler;
    private int subsequenceLength;
    private SMO svmClassifier;

    public ShapeDTW_SVM() {
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        this.subsequenceTransformer = new Subsequences();
        this.svmClassifier = new SMO();
        this.sdtwFeats = new ShapeDTWFeatures();
        this.sampler = new RandomStratifiedSampler();
    }

    public ShapeDTW_SVM(int subsequenceLength) {
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        this.subsequenceTransformer = new Subsequences(subsequenceLength);
        this.svmClassifier = new SMO();
        this.sdtwFeats = new ShapeDTWFeatures();
        this.sampler = new RandomStratifiedSampler();
    }

    public double getTrainRefSplit() { return trainRefSplit; }
    public void setTrainRefSplit(double trainRefSplit) {this.trainRefSplit = trainRefSplit;}

    @Override
    public void buildClassifier(Instances trainInsts) throws Exception {
        // Check the data
        this.getCapabilities().testWithFail(trainInsts);
        // Record the build time.
        long buildTime = System.nanoTime();
        //Transform the trainInsts into subsequences
        Instances transformedInsts = this.subsequenceTransformer.transform(trainInsts);
        //Do a stratified sample to produce a training and reference set.
        this.
        this.sampler.setInstances(transformedInsts);
        int count = 0;
        while(this.sampler.hasNext()) {

        }
        //transform the training set into ShapeDTW features
        //build the SVM classifier on the training data
        // Store the timing results.
        buildTime = System.nanoTime() - buildTime ;
        this.trainResults.setBuildTime(buildTime);
        this.trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
    }

    @Override
    public double classifyInstance(Instance testInst) throws Exception {

    }

    @Override
    public double [] distributionForInstance(Instance testInst) throws Exception {

    }

    /**
     * Testing method for this class.
     *
     * @param args - the command line arguments.
     */
    public static void main(String[] args) throws Exception {
        Instances [] data = DatasetLoading.sampleItalyPowerDemand(0);

        ShapeDTW_SVM s = new ShapeDTW_SVM();
        s.buildClassifier(data[0]);
    }
}
