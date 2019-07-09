package distances.derivative_time_domain.ddtw;

import distances.time_domain.dtw.Dtw;
import evaluation.tuning.ParameterSpace;
import timeseriesweka.filters.DerivativeFilter;
import utilities.ArrayUtilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

public class Ddtw extends Dtw {

    private final Filter derivative = new DerivativeFilter();

    @Override
    public double distance(Instance a,
                           Instance b,
                           final double cutOff) {
        Instances instances = new Instances(a.dataset(), 0);
        instances.add(a);
        instances.add(b);
        try {
            instances = Filter.useFilter(instances, derivative);
            a = instances.get(0);
            b = instances.get(1);
            return super.distance(a, b, cutOff);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    public static final String NAME = "DDTW";

    @Override
    public String toString() {
        return NAME;
    }

    public static ParameterSpace warpParameterSpace(Instances instances) {
        ParameterSpace parameterSpace = Dtw.warpParameterSpace(instances);
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, new String[] {NAME});
        return parameterSpace;
    }

    public static ParameterSpace edParameterSpace() {
        ParameterSpace parameterSpace = Dtw.edParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, new String[] {NAME});
        return parameterSpace;
    }

    public static ParameterSpace fullWarpParameterSpace() {
        ParameterSpace parameterSpace = Dtw.fullWarpParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, new String[] {NAME});
        return parameterSpace;
    }

    public static ParameterSpace allWarpParameterSpace(Instances instances) {
        ParameterSpace parameterSpace = new ParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, new String[] {NAME});
        parameterSpace.addParameter(WARPING_WINDOW_KEY, ArrayUtilities.incrementalRange(0, instances.numAttributes() - 1, 101));
        return parameterSpace;
    }
}
