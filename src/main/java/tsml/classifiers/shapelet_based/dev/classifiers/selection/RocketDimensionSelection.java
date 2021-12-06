package tsml.classifiers.shapelet_based.dev.classifiers.selection;

import evaluation.evaluators.SingleTestSetEvaluator;
import evaluation.storage.ClassifierResults;
import experiments.Experiments;
import tsml.classifiers.kernel_based.ROCKETClassifier;
import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.classifiers.shapelet_based.dev.filter.RandomFilter;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;
import tsml.transformers.ROCKET;
import weka.classifiers.Classifier;

import java.util.ArrayList;
import java.util.List;

public class RocketDimensionSelection extends ElbowSelection {




    protected ArrayList<DimensionResult> getDimensionResults(TimeSeriesInstances data) throws Exception{
        ArrayList<DimensionResult> dimensionResults = new ArrayList<DimensionResult>();
        int m = data.getMinLength()-1;


        for (int i=0;i<this.numDimensions;i++){
            TimeSeriesInstances dimensionInstances = new TimeSeriesInstances(data.getHSliceArray(new int[]{i}),data.getClassIndexes(), data.getClassLabels());

            ROCKETClassifier rocket = new ROCKETClassifier();
            rocket.buildClassifier(dimensionInstances);
            SingleTestSetEvaluator eval = new SingleTestSetEvaluator();
            ClassifierResults results =  eval.evaluate(rocket,dimensionInstances);

            System.out.println("Dimension " + i + " avg: " + results.getAcc());
            dimensionResults.add(new DimensionResult(i,results.getAcc()));

        }
        return dimensionResults;
    }


}
