package distances.derivative_time_domain.wddtw;

import distances.time_domain.wdtw.Wdtw;
import evaluation.tuning.ParameterSpace;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

import static distances.derivative_time_domain.Derivative.DERIVATIVE_FILTER;

public class CachedWddtw extends Wdtw {

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

    public static final String NAME = "CWDDTW";

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
