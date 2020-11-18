package tsml.data_containers.utilities;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import tsml.data_containers.TimeSeries;

public class TimeSeriesSummaryStatistics {

    private double mean;
    private double sum;
    private double slope;
    private double variance;
    private double kurtosis;
    private double min;
    private double max;
    private double sumSq;
    private double skew;
    private double std;

    public TimeSeriesSummaryStatistics(double[] data) {
        // this method assume that there is no NaNs present.
        // use with care.
        calculateStats(data);
    }

    public TimeSeriesSummaryStatistics(TimeSeries data) {
        this(data.getSeries());
    }

    public TimeSeriesSummaryStatistics(List<Double> data) {
        // calculate stats
        // strip out the NaNs, convert to an array of doubles.
        this(convert(data));
    }

    
    /** 
     * @param inst
     */
    public void calculateStats(double[] inst) {
        max = max(inst);
        min = min(inst);
        sum = sum(inst);
        sumSq = sumSq(inst);

        mean = mean(inst);
        variance = variance(inst, mean);

        std = Math.sqrt(variance);

        skew = skew(inst, mean, std);
        kurtosis = kurtosis(inst, mean, std);
        slope = slope(inst, sum, sumSq, std);
    }

    
    /** 
     * @param inst
     * @return double
     */
    /* Surprised these don't exist */
    public static double sum(double[] inst) {
        double sumSq = 0;
        for (double x : inst) {
            sumSq += x;
        }
        return sumSq;
    }

    
    /** 
     * @param data
     * @return double
     */
    public static double sum(List<Double> data){
        return sum(convert(data));
    }

    
    /** 
     * @param ts
     * @return double
     */
    public static double sum(TimeSeries ts){
        return sum(ts.getSeries());
    }

    
    /** 
     * @param inst
     * @return double
     */
    public static double sumSq(double[] inst) {
        double sumSq = 0;
        for (double x : inst) {
            sumSq += x * x;
        }
        return sumSq;
    }

    
    /** 
     * @param data
     * @return double
     */
    public static double sumSq(List<Double> data){
        return sumSq(convert(data));
    }

    
    /** 
     * @param ts
     * @return double
     */
    public static double sumSq(TimeSeries ts){
        return sumSq(ts.getSeries());
    }

    
    /** 
     * @param inst
     * @return int
     */
    public static int argmax(double[] inst) {
        double max = Double.MIN_VALUE;
        int arg = -1;
        int j = 0;
        for (double x : inst) {
            if (x > max) {
                max = x;
                arg = j;
            }
            j++;
        }
        return arg;
    }

    
    /** 
     * @param data
     * @return int
     */
    public static int argmax(List<Double> data){
        return argmax(convert(data));
    }

    
    /** 
     * @param ts
     * @return int
     */
    public static int argmax(TimeSeries ts){
        return argmax(ts.getSeries());
    }

    
    /** 
     * @param inst
     * @return double
     */
    public static double max(double[] inst) {
        return inst[argmax(inst)];
    }

    
    /** 
     * @param data
     * @return double
     */
    public static double max(List<Double> data){
        return max(convert(data));
    }

    
    /** 
     * @param ts
     * @return double
     */
    public static double max(TimeSeries ts){
        return max(ts.getSeries());
    }

    
    /** 
     * @param inst
     * @return int
     */
    public static int argmin(double[] inst) {
        double min = Double.MAX_VALUE;
        int arg = -1;
        int j = 0;
        for (double x : inst) {
            if (x < min) {
                min = x;
                arg = j;
            }
            j++;
        }
        return arg;
    }

    
    /** 
     * @param data
     * @return int
     */
    public static int argmin(List<Double> data){
        return argmin(convert(data));
    }

    
    /** 
     * @param ts
     * @return int
     */
    public static int argmin(TimeSeries ts){
        return argmin(ts.getSeries());
    }

    
    /** 
     * @param inst
     * @return double
     */
    public static double min(double[] inst) {
        return inst[argmin(inst)];
    }

    
    /** 
     * @param data
     * @return double
     */
    public static double min(List<Double> data){
        return min(convert(data));
    }

    
    /** 
     * @param ts
     * @return double
     */
    public static double min(TimeSeries ts){
        return min(ts.getSeries());
    }

    
    /** 
     * @param inst
     * @return double
     */
    public static double mean(double[] inst) {
        double mean = 0;
        for (double x : inst)
            mean += x;
        return mean / (double) (inst.length);
    }

    
    /** 
     * @param data
     * @return double
     */
    public static double mean(List<Double> data){
        return mean(convert(data));
    }

    
    /** 
     * @param ts
     * @return double
     */
    public static double mean(TimeSeries ts){
        return mean(ts.getSeries());
    }

    
    /** 
     * @param inst
     * @param mean
     * @return double
     */
    public static double variance(double[] inst, double mean) {
        double var = 0;
        for (double x : inst)
            var += Math.pow(x - mean, 2);
        return var / (double) (inst.length);
    }

    
    /** 
     * @param data
     * @param mean
     * @return double
     */
    public static double variance(List<Double> data, double mean){
        return variance(convert(data), mean);
    }

    
    /** 
     * @param ts
     * @param mean
     * @return double
     */
    public static double variance(TimeSeries ts, double mean){
        return variance(ts.getSeries(), mean);
    }

    
    /** 
     * @param inst
     * @param mean
     * @param std
     * @return double
     */
    public static double kurtosis(double[] inst, double mean, double std) {
        double kurt = 0;
        for (double x : inst)
            kurt += Math.pow(x - mean, 4);

        kurt /= Math.pow(std, 4);
        return kurt / (double) (inst.length);
    }

    
    /** 
     * @param data
     * @param mean
     * @param std
     * @return double
     */
    public static double kurtosis(List<Double> data, double mean, double std){
        return kurtosis(convert(data), mean, std);
    }

    
    /** 
     * @param ts
     * @param mean
     * @param std
     * @return double
     */
    public static double kurtosis(TimeSeries ts, double mean, double std){
        return kurtosis(ts.getSeries(), mean, std);
    }

    
    /** 
     * @param inst
     * @param mean
     * @param std
     * @return double
     */
    public static double skew(double[] inst, double mean, double std) {
        double skew = 0;
        for (double x : inst)
            skew += Math.pow(x - mean, 3);
        skew /= Math.pow(std, 3);
        return skew / (double) (inst.length);
    }

    
    /** 
     * @param data
     * @param mean
     * @param std
     * @return double
     */
    public static double skew(List<Double> data, double mean, double std){
        return skew(convert(data), mean, std);
    }

    
    /** 
     * @param ts
     * @param mean
     * @param std
     * @return double
     */
    public static double skew(TimeSeries ts, double mean, double std){
        return skew(ts.getSeries(), mean, std);
    }

    
    /** 
     * @param inst
     * @param sum
     * @param sumSq
     * @param std
     * @return double
     */
    public static double slope(double[] inst, double sum, double sumSq, double std) {
        double sumXY = 0;
        for (int j = 0; j < inst.length; j++) {
            sumXY += inst[j] * j;
        }
        double length = inst.length;

        double sqsum = sum * sum;
        // slope
        double slope = sumXY - sqsum / length;
        double denom = sumSq - sqsum / length;
        if (denom != 0)
            slope /= denom;
        else
            slope = 0;

        return std != 0 ? slope : 0;
    }

    
    /** 
     * @param data
     * @param sum
     * @param sumSq
     * @param std
     * @return double
     */
    public static double slope(List<Double> data, double sum, double sumSq, double std){
        return slope(convert(data), sum, sumSq, std);
    }

    
    /** 
     * @param ts
     * @param sum
     * @param sumSq
     * @param std
     * @return double
     */
    public static double slope(TimeSeries ts, double sum, double sumSq, double std){
        return slope(ts.getSeries(), sum, sumSq, std);
    }


    
    /** 
     * @param data
     * @param min
     * @param max
     * @return List<Double>
     */
    public static List<Double> intervalNorm(List<Double> data, double min, double max){
        return convert(intervalNorm(convert(data), min, max));
    }

    
    /** 
     * @param ts
     * @param min
     * @param max
     * @return TimeSeries
     */
    public static TimeSeries intervalNorm(TimeSeries ts, double min, double max){
        return new TimeSeries(intervalNorm(ts.toValueArray(), min, max));
    }

    
    /** 
     * @param data
     * @param min
     * @param max
     * @return double[]
     */
    public static double[] intervalNorm(double[] data, double min, double max){
        double[] out = new double[data.length];
        for(int i=0; i<out.length; i++)
            out[i] = (data[i] - min) / (max - min);
            
        return out;
    }
    
    
    /** 
     * @param data
     * @param mean
     * @param std
     * @return List<Double>
     */
    public static List<Double> standardNorm(List<Double> data, double mean, double std){
        return convert(standardNorm(convert(data), mean, std));
    }

    
    /** 
     * @param ts
     * @param mean
     * @param std
     * @return TimeSeries
     */
    public static TimeSeries standardNorm(TimeSeries ts, double mean, double std){
        return new TimeSeries(standardNorm(ts.toValueArray(), mean, std));
    }

    
    /** 
     * @param ts
     * @return TimeSeries
     */
    public static TimeSeries standardNorm(TimeSeries ts){
        double mean = mean(ts);
        double std = Math.sqrt(variance(ts, mean));
        return new TimeSeries(standardNorm(ts.toValueArray(), mean, std));
    }

    
    /** 
     * @param data
     * @param mean
     * @param std
     * @return double[]
     */
    public static double[] standardNorm(double[] data, double mean, double std){
        double[] out = new double[data.length];
        for(int i=0; i<out.length; i++)
            out[i] =  (data[i] - mean) / (std);
            
        return out;
    }

    
    /** 
     * @return double
     */
    public double getMean() {
        return mean;
    }

    
    /** 
     * @return double
     */
    public double getSum() {
        return sum;
    }

    
    /** 
     * @return double
     */
    public double getSlope() {
        return slope;
    }

    
    /** 
     * @return double
     */
    public double getVariance() {
        return variance;
    }

    
    /** 
     * @return double
     */
    public double getKurtosis() {
        return kurtosis;
    }

    
    /** 
     * @return double
     */
    public double getMin() {
        return min;
    }

    
    /** 
     * @return double
     */
    public double getMax() {
        return max;
    }

    
    /** 
     * @return double
     */
    public double getSumSq() {
        return sumSq;
    }

    
    /** 
     * @return double
     */
    public double getSkew() {
        return skew;
    }


    
    /** 
     * @param in
     * @return double[]
     */
    private static double[] convert(List<Double> in){
        return in.stream().filter(Double::isFinite).mapToDouble(Double::doubleValue).toArray();
    }

    
    /** 
     * @param in
     * @return List<Double>
     */
    private static List<Double> convert(double[] in){
        return DoubleStream.of(in).boxed().collect(Collectors.toList());
    }

}
