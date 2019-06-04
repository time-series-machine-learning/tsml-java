package distances.ddtw;

import distances.dtw.Dtw;
import timeseriesweka.filters.DerivativeFilter;

public class Ddtw extends Dtw {

    @Override
    protected double measureDistance(final double[] timeSeriesA, final double[] timeSeriesB, final double cutOff) {
        return super.measureDistance(DerivativeFilter.derivative(timeSeriesA), DerivativeFilter.derivative(timeSeriesB), cutOff);
    }

}
