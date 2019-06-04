package distances.wddtw;

import distances.wdtw.Wdtw;
import timeseriesweka.filters.DerivativeFilter;

public class Wddtw extends Wdtw {
    @Override
    protected double measureDistance(final double[] a, final double[] b, final double cutOff) {
        return super.measureDistance(DerivativeFilter.derivative(a), DerivativeFilter.derivative(b), cutOff);
    }

}
