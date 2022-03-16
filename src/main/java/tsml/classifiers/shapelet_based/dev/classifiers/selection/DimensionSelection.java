package tsml.classifiers.shapelet_based.dev.classifiers.selection;

import evaluation.evaluators.SingleTestSetEvaluatorTS;
import experiments.Experiments;
import fileIO.OutFile;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import weka.core.Capabilities;
import weka.core.Instances;

import java.io.File;
import java.util.Arrays;
import java.util.Random;

public abstract class DimensionSelection extends EnhancedAbstractClassifier implements TrainTimeContractable {

    private boolean normalise = true;
    private long trainContractTimeNanos = 0;
    private boolean trainTimeContract = false;

    Random rand = new Random();

    protected MSTC classifier;
    protected Experiments.ExperimentalArguments exp;
    protected MSTC.ShapeletParams params;
    protected int[] indexes;
    protected int numDimensions;

    public DimensionSelection( Experiments.ExperimentalArguments exp, MSTC.ShapeletParams params){

        this.exp = exp;
        this.params = params;
        this.estimateOwnPerformance = true;

    }

    abstract int[] getIndexes(TimeSeriesInstances data) throws Exception;




    @Override
    public String getParameters(){
        return classifier.filter.getParameters() + ", Num Dimensions, " + this.indexes.length + "/" + this.numDimensions+", Dimension Selected," +    Arrays.toString(this.indexes);
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
        classifier = new MSTC(data.numClasses(), exp, params);
        this.numDimensions = data.getMaxNumDimensions();
        this.indexes = getIndexes(data);
        saveData(this.indexes);

        TimeSeriesInstances finalData = new TimeSeriesInstances(data.getHSliceArray(this.indexes),data.getClassIndexes(), data.getClassLabels());
        classifier.buildClassifier(finalData);
        SingleTestSetEvaluatorTS eval = new SingleTestSetEvaluatorTS(this.exp.foldId, false, true, this.exp.interpret); //DONT clone data, DO set the class to be missing for each inst

        this.trainResults =  eval.evaluate(classifier, data);
        this.trainResults.setParas(getParameters());

    }

    private void saveData(int[] indexes) throws Exception {
        OutFile out = null;
        try {
            File directory = new File(exp.resultsWriteLocation + "/"+ exp.classifierName + "/ds/" + exp.datasetName);
            if (! directory.exists()){
                directory.mkdir();
                // If you require it to make the entire directory path including parents,
                // use directory.mkdirs(); here instead.
            }

            out = new OutFile(exp.resultsWriteLocation + "/"+ exp.classifierName + "/ds/" + exp.datasetName + "/ds" + exp.foldId + ".csv");
            out.writeString(Arrays.toString(indexes).replace("[","").replace("]",""));
        } catch (Exception e) {
            throw new Exception("Error writing file.\n"
                    + "Outfile most likely didnt open successfully, probably directory doesnt exist yet.\n"
                    +"\nError: "+ e);
        } finally {
            if (out != null)
                out.closeFile();
        }
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
