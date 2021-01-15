package tsml.classifiers.distance_based.distances.twed.spaces;

import tsml.classifiers.distance_based.distances.twed.TWEDistance;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.data_containers.TimeSeriesInstances;

import java.util.List;

import static utilities.ArrayUtilities.unique;

public class TWEDistanceParams implements ParamSpaceBuilder {
    @Override public ParamSpace build(final TimeSeriesInstances data) {
        double[] nuValues = {
                // <editor-fold defaultstate="collapsed" desc="hidden for space">
                0.00001,
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.5,
                1,// </editor-fold>
        };
        double[] lambdaValues = {
                // <editor-fold defaultstate="collapsed" desc="hidden for space">
                0,
                0.011111111,
                0.022222222,
                0.033333333,
                0.044444444,
                0.055555556,
                0.066666667,
                0.077777778,
                0.088888889,
                0.1,// </editor-fold>
        };
        List<Double> nuValuesUnique = unique(nuValues);
        List<Double> lambdaValuesUnique = unique(lambdaValues);
        ParamSpace params = new ParamSpace();
        params.add(TWEDistance.LAMBDA_FLAG, lambdaValuesUnique);
        params.add(TWEDistance.NU_FLAG, nuValuesUnique);
        return params;
    }
}
