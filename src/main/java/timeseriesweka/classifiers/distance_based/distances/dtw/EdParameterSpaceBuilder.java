package timeseriesweka.classifiers.distance_based.distances.dtw;

import evaluation.tuning.ParameterSpace;
import evaluation.tuning.ParameterSpaceBuilder;

import static timeseriesweka.classifiers.distance_based.distances.DistanceMeasure.DISTANCE_MEASURE_KEY;
import static timeseriesweka.classifiers.distance_based.distances.dtw.Dtw.WARPING_WINDOW_KEY;

public class EdParameterSpaceBuilder extends ParameterSpaceBuilder {
    @Override
    public ParameterSpace build() {
        ParameterSpace parameterSpace = new ParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, Dtw.NAME);
        parameterSpace.addParameter(WARPING_WINDOW_KEY, 0);
        return parameterSpace;
    }
}
