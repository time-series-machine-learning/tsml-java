package tsml.classifiers.shapelet_based.dev.classifiers.selection;

import evaluation.evaluators.SingleTestSetEvaluatorTS;
import evaluation.storage.ClassifierResults;
import experiments.ExperimentsTS;
import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.data_containers.TimeSeriesInstances;

public class RandomDimensionSelectionMSTC extends DimensionSelectionMSTC {

    private int k=5;

    public RandomDimensionSelectionMSTC(ExperimentsTS.ExperimentalArguments exp, MSTC.ShapeletParams params) {
        super(exp, params);
    }

    private int[] getRandomIndexes(){
        int[] ind = new int[k];
        for  (int i=0;i<k;i++){
            ind[i] = rand.nextInt(this.numDimensions);
        }
        return ind;
    }

    int[] getIndexes(TimeSeriesInstances data) throws Exception{
        double acc = -9999;
        int[] besIndexes = new int[this.k];
        for (int i=0;i<100;i++){
            int[] ind = getRandomIndexes();
            TimeSeriesInstances currentData = new TimeSeriesInstances(data.getHSliceArray(ind),data.getClassIndexes(), data.getClassLabels());
            MSTC.ShapeletParams currentParams = new MSTC.ShapeletParams(this.params);
            currentParams.maxIterations = 1000;
            currentParams.classifier = MSTC.AuxClassifiers.LINEAR;
            currentParams.k = 100;
            currentParams.allowZeroQuality = true;
            MSTC currentClassifier = new MSTC(currentParams);
            SingleTestSetEvaluatorTS eval = new SingleTestSetEvaluatorTS(1, false, true, exp.interpret); //DONT clone data, DO set the class to be missing for each inst
            ClassifierResults results = eval.evaluate(currentClassifier,currentData,currentData);
            if (results.getAcc()>acc){
                besIndexes = ind;
                acc = results.getAcc();
            }
        }
        return besIndexes;
    }
}
