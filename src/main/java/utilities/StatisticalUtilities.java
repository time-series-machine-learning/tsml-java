/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package utilities;

import weka.core.Instance;
import weka.core.Instances;

import java.math.BigDecimal;
import java.math.MathContext;
import java.util.*;

import static java.math.BigDecimal.ROUND_HALF_UP;

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


    

    public static double pStdDev(Instances input){
        if(input.classIndex() != input.numAttributes() - 1) {
            throw new IllegalArgumentException("class value must be at the end");
        }
        double sumx = 0;
        double sumx2 = 0;
        double[] ins2array;
        for(int i = 0; i < input.numInstances(); i++){
            final Instance instance = input.instance(i);
            for(int j = 0; j < instance.numAttributes()-1; j++){//-1 to avoid classVal
                final double value = instance.value(j);
                sumx+= value;
                sumx2+= value * value;
            }
        }
        int n = input.numInstances()*(input.numAttributes()-1);
        double mean = sumx/n;
        return Math.sqrt(sumx2/(n)-mean*mean);

    }



    // jamesl
    // the median of a list of values, just sorts (a copy, original remains unsorted) and takes middle for now
    // can make O(n) if wanted later
    public static double median(double[] values) { return median(values, true); }

    public static double median(double[] values, boolean copyArr) {
        double[] copy;
        if (copyArr) copy = Arrays.copyOf(values, values.length); else copy = values;
        Arrays.sort(copy);
        if (copy.length % 2 == 1)
            return copy[copy.length/2];
        else 
            return (copy[copy.length/2 - 1] + copy[copy.length/2]) / 2;
    }

    public static double median(ArrayList<Double> values) { return median(values, true); }

    public static double median(ArrayList<Double> values, boolean copyArr) {
        ArrayList<Double> copy;
        if (copyArr) copy = new ArrayList<>(values); else copy = values;
        Collections.sort(copy);
        if (copy.size() % 2 == 1)
            return copy.get(copy.size()/2);
        else
            return (copy.get(copy.size()/2 - 1) + copy.get(copy.size()/2)) / 2;
    }

    public static double standardDeviation(double[] values, boolean classVal) {
        double mean = mean(values, classVal);
        double sumSquaresDiffs = 0;
        int offset = classVal ? 1 : 0;

        for (int i = 0; i < values.length - offset; i++) {
            double diff = values[i] - mean;

            sumSquaresDiffs += diff * diff;
        }

        return Math.sqrt(sumSquaresDiffs / (values.length - 1 - offset));
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
            normalizedVector[i] = !NumUtils.isNearlyEqual(std, 0.0) ? (vector[i] - mean) / std : 0;
        }

        return normalizedVector;
    }
    
    public static double[] normalize(double[] vector) {
        return StatisticalUtilities.normalize(vector, false);
    }
    

    //Aaron: I'm not confident in the others...I may have written those too... lol
    public static double[] norm(double[] patt){
        double mean =0,sum =0 ,sumSq =0,var = 0;
        for(int i=0; i< patt.length; ++i){
            sum = patt[i];
            sumSq = patt[i]*patt[i];
        }

        double size= patt.length;
		var = (sumSq - sum * sum / size) / size;
        mean = sum / size;
        
        double[] out = new double[patt.length];
        if(NumUtils.isNearlyEqual(var, 0.0)){
            for(int i=0; i<patt.length; ++i)
                out[i] = 0.0;
        }
        else{
            double stdv = Math.sqrt(var);
            for(int i=0; i<patt.length; ++i)
                out[i] = (patt[i] - mean) / stdv;
        }

        return out;
    }

    public static void normInPlace(double[] r){
        double sum=0,sumSq=0,mean=0,stdev=0;
        for(int i=0;i<r.length;i++){
                sum+=r[i];
                sumSq+=r[i]*r[i];
        }
        stdev=(sumSq-sum*sum/r.length)/r.length;
        mean=sum/r.length;
        if(stdev==0){
            for (int j = 0; j < r.length; ++j)
                r[j] = 0;
        }
        else{
            stdev=Math.sqrt(stdev);
            for(int i=0;i<r.length;i++)
                r[i]=(r[i]-mean)/stdev;
        }
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

    public static BigDecimal sqrt(BigDecimal decimal) {
        int scale = MathContext.DECIMAL128.getPrecision();
        BigDecimal x0 = new BigDecimal("0");
        BigDecimal x1 = BigDecimal.valueOf(Math.sqrt(decimal.doubleValue()));
        while (!x0.equals(x1)) {
            x0 = x1;
            x1 = decimal.divide(x0, scale, ROUND_HALF_UP);
            x1 = x1.add(x0);
            x1 = x1.divide(TWO, scale, ROUND_HALF_UP);
        }
        return x1;
    }

    public static double dot(double[] inst1, double[] inst2){
        double sum = 0;
        for (int i = 0; i < inst1.length; i++)
            sum += inst1[i] * inst2[i];
        return sum;
    }

    public static int dot(int[] inst1, int[] inst2){
        int sum = 0;
        for (int i = 0; i < inst1.length; i++)
            sum += inst1[i] * inst2[i];
        return sum;
    }
}
