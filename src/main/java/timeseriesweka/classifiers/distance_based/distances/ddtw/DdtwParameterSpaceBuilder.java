package timeseriesweka.classifiers.distance_based.distances.ddtw;

import evaluation.tuning.ParameterSpace;
import timeseriesweka.classifiers.distance_based.distances.dtw.Dtw;
import timeseriesweka.classifiers.distance_based.distances.dtw.DtwParameterSpaceBuilder;
import timeseriesweka.classifiers.distance_based.distances.wddtw.Wddtw;

import static timeseriesweka.classifiers.distance_based.distances.DistanceMeasure.DISTANCE_MEASURE_KEY;

public class DdtwParameterSpaceBuilder extends DtwParameterSpaceBuilder {

    @Override
    public ParameterSpace build() {
        ParameterSpace parameterSpace = super.build();
        parameterSpace.putParameter(DISTANCE_MEASURE_KEY, Ddtw.NAME);
        return parameterSpace;
    }
}
