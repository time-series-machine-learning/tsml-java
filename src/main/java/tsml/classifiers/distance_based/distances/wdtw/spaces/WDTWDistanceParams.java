package tsml.classifiers.distance_based.distances.wdtw.spaces;

import tsml.classifiers.distance_based.distances.wdtw.WDTW;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.data_containers.TimeSeriesInstances;

import java.util.List;

import static utilities.ArrayUtilities.unique;

public class WDTWDistanceParams implements ParamSpaceBuilder {

    @Override public ParamSpace build(final TimeSeriesInstances data) {
        double[] gValues = new double[101];
        for(int i = 0; i < gValues.length; i++) {
            gValues[i] = (double) i / 100;
        }
        List<Double> gValuesUnique = unique(gValues);
        ParamSpace params = new ParamSpace();
        params.add(WDTW.G_FLAG, gValuesUnique);
        return params;
    }
}
