package tsml.classifiers.shapelet_based.dev.classifiers.selection;

import experiments.ExperimentsTS;
import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.data_containers.TimeSeriesInstances;

public class RandomDimensionSelectionMSTC extends DimensionSelection {

    private int k=5;

    public RandomDimensionSelectionMSTC(int numClasses, ExperimentsTS.ExperimentalArguments exp, MSTC.ShapeletParams params) {
        super(numClasses, exp, params);
    }

    private int[] getRandomIndexes(){
        int[] ind = new int[k];
        for  (int i=0;i<k;i++){
            ind[i] = rand.nextInt(this.numDimensions);
        }
        return ind;
    }

    int[] getIndexes(TimeSeriesInstances data) throws Exception{
        //double acc = -9999;
        this.numDimensions = data.getMaxNumDimensions();
        this.k = getRandom().nextInt(this.numDimensions)+1;
        return getRandomIndexes();
      /*  for (int i=0;i<100;i++){
            int[] ind = getRandomIndexes();
            TimeSeriesInstances currentData = new TimeSeriesInstances(data.getHSliceArray(ind),data.getClassIndexes(), data.getClassLabels());
            MSTC.ShapeletParams currentParams = new MSTC.ShapeletParams(this.params);
            currentParams.maxIterations = 1000;
            currentParams.classifier = MSTC.AuxClassifiers.LINEAR;
            currentParams.k = 100;
            currentParams.allowZeroQuality = true;
            MSTC currentClassifier = new MSTC(this.numClasses, this.exp, currentParams);
            SingleTestSetEvaluatorTS eval = new SingleTestSetEvaluatorTS(1, false, true, exp.interpret); //DONT clone data, DO set the class to be missing for each inst
            ClassifierResults results = eval.evaluate(currentClassifier,currentData,currentData);
            if (results.getAcc()>acc){
                besIndexes = ind;
                acc = results.getAcc();
            }
        }
        return besIndexes;*/
    }
}
