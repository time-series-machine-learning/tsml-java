package timeseriesweka.measures;

import utilities.SaveParameterInfo;
import weka.core.Instance;
import weka.core.NormalizableDistance;
import weka.core.TechnicalInformationHandler;

import static utilities.Utilities.extractTimeSeries;
import static utilities.Utilities.notNullCheck;
import static utilities.Utilities.positiveCheck;

// todo summary for each measure / relate to paper
// auth
// d Itakura Parallelogram



public abstract class DistanceMeasure extends NormalizableDistance implements SaveParameterInfo, TechnicalInformationHandler {

    public DistanceMeasure() {
        setDontNormalize(true); // disable WEKA's normalisation - shouldn't use it anyway but just in case!
    }

    @Override
    public String globalInfo() {
        throw new UnsupportedOperationException("Haven't done this yet"); // todo
    }

    @Override
    protected double updateDistance(double currDist, double diff) {
        throw new UnsupportedOperationException("Haven't done this yet"); // todo
    }

    /**
     * measures distance between time series
     * @param timeSeriesA longest time series of the two
     * @param timeSeriesB shortest time series of the two
     * @param cutOff cut off value to abandon distance measurement early
     * @return distance between two time series
     */
    protected abstract double measureDistance(double[] timeSeriesA, double[] timeSeriesB, double cutOff);

    /**
     * measures distance between time series, swapping the two time series so A is always the longest
     * @param timeSeriesA time series
     * @param timeSeriesB time series
     * @param cutOff cut off value to abandon distance measurement early
     * @return distance between two time series
     */
    public final double distance(double[] timeSeriesA, double[] timeSeriesB, double cutOff) {
        if(timeSeriesA.length < timeSeriesB.length) {
            double[] temp = timeSeriesA;
            timeSeriesA = timeSeriesB;
            timeSeriesB = temp;
        }
        return measureDistance(timeSeriesA, timeSeriesB, cutOff);
    }

    /**
     * measures distance between time series, swapping the two time series so A is always the longest
     * @param timeSeriesA time series
     * @param timeSeriesB time series
     * @return distance between two time series
     */
    public final double distance(double[] timeSeriesA, double[] timeSeriesB) {
        return distance(timeSeriesA, timeSeriesB, Double.POSITIVE_INFINITY);
    }

    /**
     * find distance between two instances
     * @param instanceA first instance
     * @param instanceB second instance
     * @return distance between the two instances
     */
    public final double distance(Instance instanceA, Instance instanceB) {
        return distance(instanceA, instanceB, Double.POSITIVE_INFINITY);
    }

    /**
     * find distance between two instances
     * @param instanceA first instance
     * @param instanceB second instance
     * @param cutOff cut off value to abandon distance measurement early
     * @return distance between the two instances
     */
    public final double distance(Instance instanceA, Instance instanceB, double cutOff) {
        return measureDistance(extractTimeSeries(instanceA), extractTimeSeries(instanceB), cutOff);
    }
}
