package evaluation.storage;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

public abstract class EstimatorResults {

    //LINE 1: meta info, set by user

    protected String datasetName = "";

    protected String split = ""; //e.g train or test

    protected int foldID = -1;

    protected String description= ""; //human-friendly optional extra info if wanted.

    //LINE 3: acc, buildTime, testTime, memoryUsage

    /**
     * The time taken to complete the build of an estimator, aka training. May be cumulative time over many parameter
     * set builds, etc It is assumed that the time given will be in the unit of measurement set by this object TimeUnit,
     * default nanoseconds. If no benchmark time is supplied, the default value is -1
     */
    protected long buildTime = -1;

    /**
     * The cumulative prediction time, equal to the sum of the individual prediction times stored. Intended as a quick
     * helper/summary in case complete prediction information is not stored, and/or for a human reader to quickly
     * compare times.
     *
     * It is assumed that the time given will be in the unit of measurement set by this object TimeUnit,
     * default nanoseconds.
     * If no benchmark time is supplied, the default value is -1
     */
    protected long testTime = -1;
    protected long medianPredTime;

    /**
     * The time taken to perform some standard benchmarking operation, to allow for a (not necessarily precise)
     * way to measure the general speed of the hardware that these results were made on, such that users
     * analysing the results may scale the timings in this file proportional to the benchmarks to get a consistent
     * relative scale across different results sets. It is up to the user what this benchmark operation is, and how
     * long it is (roughly) expected to take.
     * <p>
     * It is assumed that the time given will be in the unit of measurement set by this object TimeUnit, default
     * nanoseconds. If no benchmark time is supplied, the default value is -1
     */
    protected long benchmarkTime = -1;

    /**
     * It is user dependent on exactly what this field means and how accurate it may be (because of Java's lazy gc).
     * Intended purpose would be the size of the model at the end of/after estimator build, aka the estimator
     * has been trained.
     * <p>
     * The assumption, for now, is that this is measured in BYTES, but this is not enforced/ensured
     * If no memoryUsage value is supplied, the default value is -1
     */
    protected long memoryUsage = -1;

    /**
     * Consistent time unit ASSUMED across build times. Default to nanoseconds.
     * <p>
     * A long can contain 292 years worth of nanoseconds, which I assume to be enough for now.
     * Could be conceivable that the cumulative time of a large meta ensemble that is run
     * multi-threaded on a large dataset might exceed this.
     */
    protected TimeUnit timeUnit = TimeUnit.NANOSECONDS;


    public String getDatasetName() { return datasetName; }

    public void setDatasetName(String datasetName) { this.datasetName = datasetName; }

    public int getFoldID() { return foldID; }

    public void setFoldID(int foldID) { this.foldID = foldID; }

    /**
     * e.g "train", "test", "validation"
     */
    public String getSplit() { return split; }

    /**
     * e.g "train", "test", "validation"
     */
    public void setSplit(String split) { this.split = split; }

    /**
     * Consistent time unit ASSUMED across build times, test times, individual prediction times.
     * Before considering different timeunits, all timing were in milliseconds, via
     * System.currentTimeMillis(). Some classifiers on some datasets may run in less than 1 millisecond
     * however, so as of 19/2/2019, classifierResults now defaults to working in nanoseconds.
     *
     * A long can contain 292 years worth of nanoseconds, which I assume to be enough for now.
     * Could be conceivable that the cumulative time of a large meta ensemble that is run
     * multi-threaded on a large dataset might exceed this.
     *
     * In results files made before 19/2/2019, which only stored build times and
     * milliseconds was assumed, there will be no unit of measurement for the time.
     */
    public TimeUnit getTimeUnit() {
        return timeUnit;
    }

    /**
     * This will NOT convert any timings already stored in this classifier results object
     * to the new time unit. e.g if build time was had already been stored in seconds as 10, THEN
     * setTimeUnit(TimeUnit.MILLISECONDS) was called, the actual value of build time would still be 10,
     * but now assumed to mean 10 milliseconds.
     *
     * Consistent time unit ASSUMED across build times, test times, individual prediction times.
     * Before considering different timeunits, all timing were in milliseconds, via
     * System.currentTimeMillis(). Some classifiers on some datasets may run in less than 1 millisecond
     * however, so as of 19/2/2019, classifierResults now defaults to working in nanoseconds.
     *
     * A long can contain 292 years worth of nanoseconds, which I assume to be enough for now.
     * Could be conceivable that the cumulative time of a large meta ensemble that is run
     * multi-threaded on a large dataset might exceed this.
     *
     * In results files made before 19/2/2019, which only stored build times and
     * milliseconds was assumed, there will be no unit of measurement for the time.
     */
    public void setTimeUnit(TimeUnit timeUnit) {
        this.timeUnit = timeUnit;
    }

    /**
     * This is a free-form description that can hold any info you want, with the only caveat
     * being that it cannot contain newline characters. Description could be the experiment
     * that these results were made for, e.g "Initial Univariate Benchmarks". Entirely
     * up to the user to process if they want to.
     *
     * By default, it is an empty string.
     */
    public String getDescription() {
        return description;
    }
    /**
     * This is a free-form description that can hold any info you want, with the only caveat
     * being that it cannot contain newline characters. Description could be the experiment
     * that these results were made for, e.g "Initial Univariate Benchmarks". Entirely
     * up to the user to process if they want to.
     *
     * By default, it is an empty string.
     */
    public void setDescription(String description) {
        this.description = description;
    }


    //todo revisit these when more willing to refactor stats pipeline to avoid assumption of doubles.
    //a double can accurately (except for the standard double precision problems) hold at most ~7 weeks worth of nano seconds
    //      a double's mantissa = 52bits, 2^52 / 1000000000 / 60 / 60 / 24 / 7 = 7.something weeks
    //so, will assume the usage/requirement for milliseconds in the stats pipeline, to avoid the potential future problem
    //of meta-ensembles taking more than a week, etc. (or even just summing e.g 30 large times to be averaged)
    //it is still preferable of course to store any timings in nano's in the classifierresults object since they'll
    //store them as longs.
    public static final Function<EstimatorResults, Double> GETTER_buildTimeDoubleMillis = (EstimatorResults cr) -> toDoubleMillis(cr.buildTime, cr.timeUnit);
    public static final Function<EstimatorResults, Double> GETTER_totalTestTimeDoubleMillis = (EstimatorResults cr) -> toDoubleMillis(cr.testTime,cr.timeUnit);
    public static final Function<EstimatorResults, Double> GETTER_avgTestPredTimeDoubleMillis = (EstimatorResults cr) -> toDoubleMillis(cr.medianPredTime, cr.timeUnit);
    public static final Function<EstimatorResults, Double> GETTER_benchmarkTime = (EstimatorResults cr) -> toDoubleMillis(cr.benchmarkTime, cr.timeUnit);
    public static final Function<EstimatorResults, Double> GETTER_buildTimeDoubleMillisBenchmarked = (EstimatorResults cr) -> divideAvoidInfinity(GETTER_buildTimeDoubleMillis.apply(cr), GETTER_benchmarkTime.apply(cr));
    public static final Function<EstimatorResults, Double> GETTER_totalTestTimeDoubleMillisBenchmarked = (EstimatorResults cr) -> divideAvoidInfinity(GETTER_totalTestTimeDoubleMillis.apply(cr), GETTER_benchmarkTime.apply(cr));
    public static final Function<EstimatorResults, Double> GETTER_avgTestPredTimeDoubleMillisBenchmarked = (EstimatorResults cr) -> divideAvoidInfinity(GETTER_avgTestPredTimeDoubleMillis.apply(cr), GETTER_benchmarkTime.apply(cr));

    public static final Function<EstimatorResults, Double> GETTER_MemoryMB = (EstimatorResults cr) -> (double)(cr.memoryUsage/1e+6);

    protected static double divideAvoidInfinity(double a, double b) {
        if(b == 0) {
            // avoid divide by 0 --> infinity
            return a;
        } else {
            return a / b;
        }
    }

    protected static double toDoubleMillis(long time, TimeUnit unit) {
        if (time < 0)
            return -1;
        if (time == 0)
            return 0;

        if (unit.equals(TimeUnit.MICROSECONDS)) {
            long pre = time / 1000;  //integer division for pre - decimal point
            long post = time % 1000;  //the remainder that needs to be converted to post decimal point, some value < 1000
            double convertedPost = (double)post / 1000; // now some fraction < 1

            return pre + convertedPost;
        }
        else if (unit.equals(TimeUnit.NANOSECONDS)) {
            long pre = time / 1000000;  //integer division for pre - decimal point
            long post = time % 1000000;  //the remainder that needs to be converted to post decimal point, some value < 1000
            double convertedPost = (double)post / 1000000; // now some fraction < 1

            return pre + convertedPost;
        }
        else {
            //not higher resolution than millis, no special conversion needed just cast to double
            return (double)unit.toMillis(time);
        }
    }

    /**
     * Makes copy of pred times to easily maintain original ordering
     */
    protected long findMedianPredTime(ArrayList<Long> predTimes) {
        List<Long> copy = new ArrayList<>(predTimes);
        Collections.sort(copy);

        int mid = copy.size()/2;
        if (copy.size() % 2 == 0)
            return (copy.get(mid) + copy.get(mid-1)) / 2;
        else
            return copy.get(mid);
    }

    public abstract double getAcc();

    public abstract void cleanPredictionInfo();

    /**
     * Will calculate all the metrics that can be found from the prediction information
     * stored in this object, UNLESS this object has been finalised (finaliseResults(..)) AND
     * has already had it's stats found (findAllStats()), e.g. if it has already been called
     * by another process.
     * <p>
     * In this latter case, this method does nothing.
     */
    public abstract void findAllStatsOnce();
}
