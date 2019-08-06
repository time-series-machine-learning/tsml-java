package timeseriesweka.classifiers.distance_based.distances.lcss;

import evaluation.tuning.ParameterSpace;
import evaluation.tuning.ParameterSpaceFromInstancesBuilder;
import utilities.ArrayUtilities;
import utilities.StatisticalUtilities;
import weka.core.Instances;

import static timeseriesweka.classifiers.distance_based.distances.DistanceMeasure.DISTANCE_MEASURE_KEY;

public class LcssParameterSpaceBuilder extends ParameterSpaceFromInstancesBuilder {

    public ParameterSpace build() {
        Instances instances = getInstances();
        double std = StatisticalUtilities.pStdDev(instances);
        double stdFloor = std*0.2;
        double[] epsilonValues = ArrayUtilities.incrementalRange(stdFloor, std, 10);
        int[] deltaValues = ArrayUtilities.incrementalRange(0, (instances.numAttributes() - 1) / 4, 10);
        ParameterSpace parameterSpace = new ParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, Lcss.NAME);
        parameterSpace.addParameter(Lcss.DELTA_KEY, deltaValues);
        parameterSpace.addParameter(Lcss.EPSILON_KEY, epsilonValues);
        return parameterSpace;
    }

}
