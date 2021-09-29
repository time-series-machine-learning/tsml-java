package tsml.classifiers.shapelet_based.dev.classifiers.selection;

import experiments.ExperimentsTS;
import tsml.classifiers.TSClassifier;
import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import weka.classifiers.Classifier;

import java.util.Arrays;
import java.util.Random;

public abstract class DimensionSelectionMSTC implements TSClassifier {

    Random rand = new Random();

    protected MSTC classifier;
    protected ExperimentsTS.ExperimentalArguments exp;
    protected MSTC.ShapeletParams params;
    protected int[] indexes;
    protected int numDimensions;

    public DimensionSelectionMSTC(ExperimentsTS.ExperimentalArguments exp, MSTC.ShapeletParams params){
        this.exp = exp;
        this.params = params;
        classifier = new MSTC(params);
    }

    abstract int[] getIndexes(TimeSeriesInstances data) throws Exception;

    @Override
    public Classifier getClassifier() {
        return null;
    }

    @Override
    public TimeSeriesInstances getTSTrainData() {
        return null;
    }

    @Override
    public void setTSTrainData(TimeSeriesInstances train) {

    }



    @Override
    public void buildClassifier(TimeSeriesInstances data) throws Exception {
        this.numDimensions = data.getMaxNumDimensions();
        this.indexes = getIndexes(data);
        System.out.println("Selected dimensions: "  + Arrays.toString(this.indexes));
        TimeSeriesInstances finalData = new TimeSeriesInstances(data.getHSliceArray(this.indexes),data.getClassIndexes(), data.getClassLabels());
        classifier.buildClassifier(finalData);

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
