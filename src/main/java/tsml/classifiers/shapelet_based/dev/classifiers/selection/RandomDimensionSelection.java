package tsml.classifiers.shapelet_based.dev.classifiers.selection;

import experiments.Experiments;
import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.data_containers.TimeSeriesInstances;

public class RandomDimensionSelection extends DimensionSelection {

    private int k=5;

    public RandomDimensionSelection(int numClasses, Experiments.ExperimentalArguments exp, MSTC.ShapeletParams params) {
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
        this.numDimensions = data.getMaxNumDimensions();
        this.k = getRandom().nextInt(this.numDimensions)+1;
        return getRandomIndexes();

    }
}
