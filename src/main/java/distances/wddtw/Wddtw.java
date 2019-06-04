package distances.wddtw;

import distances.wdtw.Wdtw;
import evaluation.tuning.ParameterSpace;
import timeseriesweka.filters.DerivativeFilter;

public class Wddtw extends Wdtw {
    @Override
    protected double measureDistance(final double[] a, final double[] b, final double cutOff) {
        return super.measureDistance(DerivativeFilter.derivative(a), DerivativeFilter.derivative(b), cutOff);
    }

    public static final String NAME = "WDTW";

    @Override
    public String toString() {
        return NAME;
    }

    public static ParameterSpace discreteParameterSpace() {
        ParameterSpace parameterSpace = Wdtw.discreteParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, new String[] {NAME});
        return parameterSpace;
    }
}
