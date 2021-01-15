package tsml.classifiers.distance_based.distances.erp.spaces;

import tsml.classifiers.distance_based.distances.erp.ERPDistance;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.data_containers.TimeSeriesInstances;
import utilities.StatisticalUtilities;

import java.util.List;

import static tsml.classifiers.distance_based.distances.dtw.DTW.WINDOW_SIZE_FLAG;
import static utilities.ArrayUtilities.range;
import static utilities.ArrayUtilities.unique;

public class ERPDistanceParams implements ParamSpaceBuilder {

    @Override public ParamSpace build(final TimeSeriesInstances data) {
        double std = StatisticalUtilities.pStdDev(data);
        double stdFloor = std * 0.2;
        double[] bandSizeValues = range(0d, 0.25, 10);
        double[] penaltyValues = range(stdFloor, std, 10);
        List<Double> penaltyValuesUnique = unique(penaltyValues);
        List<Double> bandSizeValuesUnique = unique(bandSizeValues);
        ParamSpace params = new ParamSpace();
        params.add(WINDOW_SIZE_FLAG, bandSizeValuesUnique);
        params.add(ERPDistance.G_FLAG, penaltyValuesUnique);
        return params;
    }
}
