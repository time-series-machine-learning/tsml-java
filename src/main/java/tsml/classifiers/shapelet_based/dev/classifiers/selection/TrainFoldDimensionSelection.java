package tsml.classifiers.shapelet_based.dev.classifiers.selection;

import evaluation.storage.ClassifierResults;
import experiments.Experiments;
import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;

public class TrainFoldDimensionSelection extends ElbowSelection {

    private String TRAIN_ALG = "STC";

    public TrainFoldDimensionSelection(int numClasses, Experiments.ExperimentalArguments exp, MSTC.ShapeletParams params){
        super(numClasses, exp,params);
    }

    protected ArrayList<DimensionResult> getDimensionResults(TimeSeriesInstances data) throws Exception{
        ArrayList<DimensionResult> dimensionResults = new ArrayList<DimensionResult>();

        for (int i=0;i<this.numDimensions;i++){
            ClassifierResults results = new ClassifierResults(this.exp.resultsWriteLocation + "STC/Predictions/" +
                    this.exp.datasetName + "Dimension" + (i+1) + "/trainFold" + this.exp.foldId + ".csv");
            dimensionResults.add(new DimensionResult(i,results.getAcc()));

        }
        return dimensionResults;
    }

}
