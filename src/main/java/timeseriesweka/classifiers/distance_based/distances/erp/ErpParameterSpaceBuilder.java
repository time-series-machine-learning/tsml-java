package timeseriesweka.classifiers.distance_based.distances.erp;

import evaluation.tuning.ParameterSpace;
import evaluation.tuning.ParameterSpaceFromInstancesBuilder;
import utilities.ArrayUtilities;
import utilities.StatisticalUtilities;
import weka.core.Instances;

import static timeseriesweka.classifiers.distance_based.distances.DistanceMeasure.DISTANCE_MEASURE_KEY;

public class ErpParameterSpaceBuilder extends ParameterSpaceFromInstancesBuilder {

    @Override
    public ParameterSpace build() {
        Instances instances = getInstances();
        double std = StatisticalUtilities.pStdDev(instances);
        double stdFloor = std*0.2;
        int[] bandSizeValues = ArrayUtilities.incrementalRange(0, (instances.numAttributes() - 1) / 4, 10);
        double[] penaltyValues = ArrayUtilities.incrementalRange(stdFloor, std, 10);
        ParameterSpace parameterSpace = new ParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, Erp.NAME);
        parameterSpace.addParameter(Erp.BAND_SIZE_KEY, bandSizeValues);
        parameterSpace.addParameter(Erp.PENALTY_KEY, penaltyValues);
        return parameterSpace;
    }
}
