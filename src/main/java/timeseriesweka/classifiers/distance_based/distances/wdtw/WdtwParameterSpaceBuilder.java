package timeseriesweka.classifiers.distance_based.distances.wdtw;

import evaluation.tuning.ParameterSpace;
import evaluation.tuning.ParameterSpaceBuilder;

public class WdtwParameterSpaceBuilder extends ParameterSpaceBuilder {
    @Override
    public ParameterSpace build() {
        double[] gValues = new double[100];
        for(int i = 0; i < gValues.length; i++) {
            gValues[i] = i / gValues.length;
        }
        ParameterSpace parameterSpace = new ParameterSpace();
        parameterSpace.addParameter(Wdtw.DISTANCE_MEASURE_KEY, Wdtw.NAME);
        parameterSpace.addParameter(Wdtw.WEIGHT_KEY, gValues);
        return parameterSpace;
    }
}
