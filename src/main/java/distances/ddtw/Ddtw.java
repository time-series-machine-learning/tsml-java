package distances.ddtw;

import distances.dtw.Dtw;
import distances.wdtw.Wdtw;
import evaluation.tuning.ParameterSpace;
import timeseriesweka.filters.DerivativeFilter;
import weka.core.Instances;

public class Ddtw extends Dtw {

    @Override
    protected double measureDistance(final double[] timeSeriesA, final double[] timeSeriesB, final double cutOff) {
        return super.measureDistance(DerivativeFilter.derivative(timeSeriesA), DerivativeFilter.derivative(timeSeriesB), cutOff);
    }

    public static final String NAME = "WDTW";

    @Override
    public String toString() {
        return NAME;
    }

    public static ParameterSpace discreteParameterSpace(Instances instances) {
        ParameterSpace parameterSpace = Dtw.discreteParameterSpace(instances);
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, new String[] {NAME});
        return parameterSpace;
    }

    public static ParameterSpace euclideanParameterSpace() {
        ParameterSpace parameterSpace = Dtw.euclideanParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, new String[] {NAME});
        return parameterSpace;
    }

    public static ParameterSpace fullWindowParameterSpace() {
        ParameterSpace parameterSpace = Dtw.fullWindowParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, new String[] {NAME});
        return parameterSpace;
    }
}
