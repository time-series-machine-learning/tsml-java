package timeseriesweka.classifiers.distance_based.distances.wddtw;

import evaluation.tuning.ParameterSpace;
import timeseriesweka.classifiers.distance_based.distances.wdtw.WdtwParameterSpaceBuilder;

import static timeseriesweka.classifiers.distance_based.distances.DistanceMeasure.DISTANCE_MEASURE_KEY;

public class WddtwParameterSpaceBuilder extends WdtwParameterSpaceBuilder {

    @Override
    public ParameterSpace build() {
        ParameterSpace parameterSpace = super.build();
        parameterSpace.putParameter(DISTANCE_MEASURE_KEY, Wddtw.NAME);
        return parameterSpace;
    }
}
