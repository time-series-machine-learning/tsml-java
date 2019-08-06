package timeseriesweka.classifiers.distance_based.distances.twed;

import evaluation.tuning.ParameterSpace;
import evaluation.tuning.ParameterSpaceBuilder;

import static timeseriesweka.classifiers.distance_based.distances.DistanceMeasure.DISTANCE_MEASURE_KEY;

public class TwedParameterSpaceBuilder extends ParameterSpaceBuilder {
    @Override
    public ParameterSpace build() {
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
        ParameterSpace parameterSpace = new ParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, Twed.NAME);
        parameterSpace.addParameter(Twed.NU_KEY, nuValues);
        parameterSpace.addParameter(Twed.LAMBDA_KEY, lambdaValues);
        return parameterSpace;
    }
}
