package distances.derivative_time_domain.wddtw;

import distances.time_domain.wdtw.Wdtw;
import evaluation.tuning.ParameterSpace;
import timeseriesweka.filters.DerivativeFilter;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

public class Wddtw extends Wdtw {
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

    public static final String NAME = "WDDTW";

    @Override
    public String toString() {
        return NAME;
    }

    public static ParameterSpace parameterSpace() {
        ParameterSpace parameterSpace = Wdtw.parameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, new String[] {NAME});
        return parameterSpace;
    }
}
