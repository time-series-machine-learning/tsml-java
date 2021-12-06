package tsml.classifiers.shapelet_based.dev.classifiers.selection;

import evaluation.evaluators.SingleTestSetEvaluator;
import evaluation.evaluators.SingleTestSetEvaluatorTS;
import experiments.Experiments;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.kernel_based.ROCKETClassifier;
import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import weka.core.Capabilities;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Random;

public abstract class DimensionSelection extends EnhancedAbstractClassifier implements TrainTimeContractable {

    private boolean normalise = true;
    private long trainContractTimeNanos = 0;
    private boolean trainTimeContract = false;

    Random rand = new Random();

    protected ROCKETClassifier classifier;
    protected int[] indexes;
    protected int numDimensions;

    public DimensionSelection(){

        this.estimateOwnPerformance = true;
        this.ableToEstimateOwnPerformance = true;
        classifier = new ROCKETClassifier();
    }

    abstract int[] getIndexes(TimeSeriesInstances data) throws Exception;




    @Override
    public String getParameters(){
        return "Num Dimensions, " + ((double)this.indexes.length/(double)this.numDimensions)+", Dimension Selected," +    Arrays.toString(this.indexes);
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        result.setMinimumNumberInstances(2);

        // attributes
        result.enable(Capabilities.Capability.RELATIONAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        return result;
    }


    public void setNormalise(boolean normalise) {
        this.normalise = normalise;
    }


    @Override
    public void setTrainTimeLimit(long time) {
        trainContractTimeNanos = time;
        trainTimeContract = true;
    }

    @Override
    public boolean withinTrainContract(long start) {
        if (trainContractTimeNanos <= 0) return true; //Not contracted
        return System.nanoTime() - start < trainContractTimeNanos/5*4;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        buildClassifier(Converter.fromArff(data));
    }

    @Override
    public void buildClassifier(TimeSeriesInstances data) throws Exception {
        this.numDimensions = data.getMaxNumDimensions();
        this.indexes = getIndexes(data);
        TimeSeriesInstances finalData = new TimeSeriesInstances(data.getHSliceArray(this.indexes),data.getClassIndexes(), data.getClassLabels());
        classifier.buildClassifier(finalData);
        SingleTestSetEvaluator eval = new SingleTestSetEvaluator();

        this.trainResults =  eval.evaluate(classifier, data);
        this.trainResults.setParas(getParameters());

    }

    @Override
    public double[] distributionForInstance(TimeSeriesInstance inst) throws Exception {
        TimeSeriesInstance instance = new TimeSeriesInstance(inst.getHSlice(this.indexes),inst.getLabelIndex());
        return classifier.distributionForInstance(instance);
    }

    @Override
    public double classifyInstance(TimeSeriesInstance inst) throws Exception {
        TimeSeriesInstance instance = new TimeSeriesInstance(inst.getHSlice(this.indexes),inst.getLabelIndex());
        return classifier.classifyInstance(instance);
    }

    @Override
    public double[][] distributionForInstances(TimeSeriesInstances data) throws Exception {
        TimeSeriesInstances instances  = new TimeSeriesInstances(data.getHSliceArray(this.indexes),data.getClassIndexes(), data.getClassLabels());
        return classifier.distributionForInstances(instances);

    }

    @Override
    public double[] classifyInstances(TimeSeriesInstances data) throws Exception {
        TimeSeriesInstances instances  = new TimeSeriesInstances(data.getHSliceArray(this.indexes),data.getClassIndexes(), data.getClassLabels());
        return classifier.classifyInstances(instances);
    }


}
