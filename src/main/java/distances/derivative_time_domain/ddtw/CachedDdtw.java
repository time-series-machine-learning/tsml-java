package distances.derivative_time_domain.ddtw;

import distances.time_domain.dtw.Dtw;
import evaluation.tuning.ParameterSpace;
import utilities.ArrayUtilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

import static distances.derivative_time_domain.Derivative.DERIVATIVE_FILTER;

public class CachedDdtw extends Dtw {

    @Override
    public double distance(Instance first,
                           Instance second,
                           final double cutOff) {
        Instances instances = new Instances(first.dataset(), 0);
        instances.add(first);
        instances.add(second);
        try {
            instances = Filter.useFilter(instances, DERIVATIVE_FILTER);
            first = instances.get(0);
            second = instances.get(1);
            return super.distance(first, second, cutOff);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    public static final String NAME = "CDDTW";

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

    public static ParameterSpace allDiscreteParameterSpace(Instances instances) {
        ParameterSpace parameterSpace = new ParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, new String[] {NAME});
        int[] range;
        if(instances.numAttributes() - 1 < 101) {
            range = ArrayUtilities.range(instances.numAttributes() - 1 - 1);
        } else {
            range = ArrayUtilities.incrementalRange(0, instances.numAttributes() - 1, 101);
        }
        parameterSpace.addParameter(WARPING_WINDOW_KEY, range);
        return parameterSpace;
    }
}
