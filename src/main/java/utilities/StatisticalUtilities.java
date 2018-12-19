/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities;

import java.math.BigDecimal;
import java.math.MathContext;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * A class offering statistical utility functions like the average and the
 * standard deviations
 *
 * @author aaron
 */
public class StatisticalUtilities {

    // the mean of a list of values
    public static double mean(double[] values, boolean classVal) {
        double sum = 0;
        
        int offset = classVal ? 1 : 0;

        for (int i = 0; i < values.length - offset; i++) {
            sum += values[i];
        }

        return sum / (double) (values.length - offset);
    }
    
    
    
    // jamesl
    // the median of a list of values, just sorts (a copy, original remains unsorted) and takes middle for now
    // can make O(n) if wanted later 
    public static double median(double[] values) {
        double[] copy = Arrays.copyOf(values, values.length);
        Arrays.sort(copy);
        
        if (copy .length % 2 == 1)
            return copy[copy.length/2 + 1];
        else 
            return (copy[copy.length/2] + copy[copy.length/2 + 1]) / 2;
    }

    public static double standardDeviation(double[] values, boolean classVal, double mean) {
//        double mean = mean(values, classVal);
        double sumSquaresDiffs = 0;
        int offset = classVal ? 1 : 0;

        for (int i = 0; i < values.length - offset; i++) {
            double diff = values[i] - mean;

            sumSquaresDiffs += diff * diff;
        }

        return Math.sqrt(sumSquaresDiffs / (values.length - 1 - offset));
    }
    // normalize the vector to mean 0 and std 1
    public static double[] normalize(double[] vector, boolean classVal) {
        double mean = mean(vector, classVal);
        double std = standardDeviation(vector, classVal, mean);

        double[] normalizedVector = new double[vector.length];

        for (int i = 0; i < vector.length; i++) {
            if (std != 0) {
                normalizedVector[i] = (vector[i] - mean) / std;
            }
        }

        return normalizedVector;
    }
    
    public static double[] normalize(double[] vector) {
        return StatisticalUtilities.normalize(vector, false);
    }
    
    public static void normalize2D(double[][] data, boolean classVal)
    {
        int offset = classVal ? 1 : 0;
        
        //normalise each series.
        for (double[] series : data) {
            //TODO: could make mean and STD better.
            double mean = mean(series, classVal);
            double std = standardDeviation(series, classVal,mean);
            
            //class value at the end of the series.
            for (int j = 0; j < series.length - offset; j++) { 
                if (std != 0) {
                    series[j] = (series[j] - mean) / std;
                }
            }
        }
    }

    public static double exp(double val) {
        final long tmp = (long) (1512775 * val + 1072632447);
        return Double.longBitsToDouble(tmp << 32);
    }

    public static double sumOfSquares(double[] v1, double[] v2) {
        double ss = 0;

        int N = v1.length;

        double err = 0;
        for (int i = 0; i < N; i++) {
            err = v1[i] - v2[i];
            ss += err * err;
        }

        return ss;
    }

    public static double calculateSigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    
    private static final BigDecimal TWO = BigDecimal.valueOf(2L);
 
    //calculates the square root of a bigdecimal. Give a mathcontext to specify precision.
    public static BigDecimal sqrt(BigDecimal x, MathContext mc) {
            BigDecimal g = x.divide(TWO, mc);
            boolean done = false;
            final int maxIterations = mc.getPrecision() + 1;		
            for (int i = 0; !done && i < maxIterations; i++) {
                    // r = (x/g + g) / 2
                    BigDecimal r = x.divide(g, mc);
                    r = r.add(g);
                    r = r.divide(TWO, mc);
                    done = r.equals(g);
                    g = r;
            }
            return g;
    }
/**
 * 
 * @param counts count of number of items at each level i
 * @return cumulative count of items at level <=i
 */    
    public static int[] findCumulativeCounts(int[] counts){
        int[] c=new int[counts.length];
        c[0]=counts[0];
        int i=1;
        while(i<counts.length){
            c[i]=c[i-1]+counts[i];
            i++;
        }
        return c;
    }

    
 /**
 * 
 * @param cumulativeCounts: cumulativeCounts[i] is the number of items <=i
 * as found by findCumulativeCounts 
 * cumulativeCounts[length-1] is the total number of objects
 * @return a randomly selected level i based on sample of cumulativeCounts
 */
    public static int sampleCounts(int[] cumulativeCounts, Random rand){
        int c=rand.nextInt(cumulativeCounts[cumulativeCounts.length-1]);
        int pos=0;
        while(cumulativeCounts[pos]<c)
            pos++;
        return pos;
    }
    
    
    public static double[][][][] averageFinalDimension(double[][][][][] results) { 
        double[][][][] res = new double[results.length][results[0].length][results[0][0].length][results[0][0][0].length]; 
        for (int i = 0; i < results.length; i++) 
            for (int j = 0; j < results[0].length; j++) 
                for (int k = 0; k < results[0][0].length; k++)
                    for (int l = 0; l < results[0][0][0].length; l++)
                        res[i][j][k][l] = StatisticalUtilities.mean(results[i][j][k][l], false);
        return res;
    }
    
    public static double[][][] averageFinalDimension(double[][][][] results) { 
        double[][][] res = new double[results.length][results[0].length][results[0][0].length]; 
        for (int i = 0; i < results.length; i++) 
            for (int j = 0; j < results[0].length; j++) 
                for (int k = 0; k < results[0][0].length; k++)
                    res[i][j][k] = StatisticalUtilities.mean(results[i][j][k], false);
        return res;
    }

    public static double[][] averageFinalDimension(double[][][] results) { 
        double[][] res = new double[results.length][results[0].length];           
        for (int i = 0; i < results.length; i++) 
            for (int j = 0; j < results[0].length; j++) 
                res[i][j] = StatisticalUtilities.mean(results[i][j], false);
        return res;
    }

    public static double[] averageFinalDimension(double[][] results) { 
        double[] res = new double[results.length];           
        for (int i = 0; i < results.length; i++) 
                res[i] = StatisticalUtilities.mean(results[i], false);
        return res;
    }
    
    public static double averageFinalDimension(double[] results) { 
        return StatisticalUtilities.mean(results, false);
    }
}
