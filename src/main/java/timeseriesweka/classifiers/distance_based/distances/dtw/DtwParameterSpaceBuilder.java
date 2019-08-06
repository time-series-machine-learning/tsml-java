package timeseriesweka.classifiers.distance_based.distances.dtw;

import evaluation.tuning.ParameterSpace;
import evaluation.tuning.ParameterSpaceFromInstancesBuilder;
import utilities.ArrayUtilities;

import static timeseriesweka.classifiers.distance_based.distances.DistanceMeasure.DISTANCE_MEASURE_KEY;

public class DtwParameterSpaceBuilder
    extends ParameterSpaceFromInstancesBuilder {

    @Override
    public ParameterSpace build() {
        ParameterSpace parameterSpace = new ParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, Dtw.NAME);
        parameterSpace.addParameter(Dtw.WARPING_WINDOW_KEY, ArrayUtilities.incrementalRange(0, getInstances().numAttributes() - 1, 100));
        return parameterSpace;
    }
}
