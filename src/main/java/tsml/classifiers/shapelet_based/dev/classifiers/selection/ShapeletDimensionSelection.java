package tsml.classifiers.shapelet_based.dev.classifiers.selection;

import experiments.Experiments;
import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.classifiers.shapelet_based.dev.filter.RandomFilter;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;
import java.util.List;

public class ShapeletDimensionSelection extends ElbowSelection {


    public ShapeletDimensionSelection(Experiments.ExperimentalArguments exp, MSTC.ShapeletParams params){
        super( exp,params);
    }

    protected ArrayList<DimensionResult> getDimensionResults(TimeSeriesInstances data) throws Exception{
        ArrayList<DimensionResult> dimensionResults = new ArrayList<DimensionResult>();
        int m = data.getMinLength()-1;
        MSTC.ShapeletParams params = new MSTC.ShapeletParams(100,
                3,
                Math.min(m-1,500),
                1000,1,
                MSTC.ShapeletFilters.RANDOM, MSTC.ShapeletQualities.GAIN_BINARY_FILTERED,
                MSTC.ShapeletFactories.INDEPENDENT,
                MSTC.AuxClassifiers.LINEAR);
        params.allowZeroQuality = true;
        params.removeSelfSimilar = false;

        for (int i=0;i<this.numDimensions;i++){
            TimeSeriesInstances dimensionInstances = new TimeSeriesInstances(data.getHSliceArray(new int[]{i}),data.getClassIndexes(), data.getClassLabels());

            RandomFilter rf = new RandomFilter();
            rf.setHourLimit(this.params.contractTimeHours);
            List<ShapeletMV> shapelets = rf.findShapelets(params,dimensionInstances);

            double averageQuality = shapelets.stream().mapToDouble(ShapeletMV::getQuality).average().orElse(0);
            System.out.println("Dimension " + i + " avg: " + averageQuality);
            dimensionResults.add(new DimensionResult(i,averageQuality));

        }
        return dimensionResults;
    }


}
