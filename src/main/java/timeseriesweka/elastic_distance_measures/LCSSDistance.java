/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package timeseriesweka.elastic_distance_measures;

import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;

/**
 *
 * @author sjx07ngu
 */
public class LCSSDistance extends EuclideanDistance{

    private double epsilon;
    private int delta;

    public LCSSDistance(int delta, double epsilon){
        this.m_DontNormalize = true;
        this.delta = delta;
        this.epsilon = epsilon;
    }

    public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats){
        return distance(first,second);
    }

    /**
     * distance method that converts instances to arrays of doubles
     *
     * @param first instance 1
     * @param second instance 2
     * @param cutOffValue used for early abandon
     * @return distance between instances
     */
    @Override
    public double distance(Instance first, Instance second, double cutOffValue){
//        if(this.distanceCount % 10000000 == 0){
//            System.out.println("New Instance: "+this.distanceCount);
//        }
//        this.distanceCount++;
        //remove class index from first instance if there is one
        int firtClassIndex = first.classIndex();
        double[] arr1;
        if(firtClassIndex > 0){
            arr1 = new double[first.numAttributes()-1];
            for(int i = 0,j = 0; i < first.numAttributes(); i++){
                if(i != firtClassIndex){
                    arr1[j]= first.value(i);
                    j++;
                }
            }
        }else{
            arr1 = first.toDoubleArray();
        }

        //remove class index from second instance if there is one
        int secondClassIndex = second.classIndex();
        double[] arr2;
        if(secondClassIndex > 0){
            arr2 = new double[second.numAttributes()-1];
            for(int i = 0,j = 0; i < second.numAttributes(); i++){
                if(i != secondClassIndex){
                    arr2[j]= second.value(i);
                    j++;
                }
            }
        }else{
            arr2 = second.toDoubleArray();
        }

        return distance(arr1,arr2,cutOffValue);
    }

    /**
     * calculates the distance between two instances (been converted to arrays)
     *
     * @param first instance 1 as array
     * @param second instance 2 as array
     * @param cutOffValue used for early abandon
     * @return distance between instances
     */
    public double distance(double[] first, double[] second, double cutOffValue){
        return distance(first, second);
    }

    public double distance(double[] first, double[] second){

        double[] a  = first;
        double[] b = second;
        int m = first.length;
        int n = second.length;

        int[][] lcss = new int[m+1][n+1];
        int[][] lastX = new int[m+1][n+1];
        int[][] lastY = new int[m+1][n+1];


        for(int i = 0; i < m; i++){
            for(int j = i-delta; j <= i+delta; j++){
//                System.out.println("here");
                if(j < 0 || j >= n){
                    //do nothing
                }else if(b[j]+this.epsilon >= a[i] && b[j]-epsilon <=a[i]){
                    lcss[i+1][j+1] = lcss[i][j]+1;
                    lastX[i+1][j+1] = i;
                    lastY[i+1][j+1] = j;
                }else if(lcss[i][j+1] > lcss[i+1][j]){
                    lcss[i+1][j+1] = lcss[i][j+1];
                    lastX[i+1][j+1] = i;
                    lastY[i+1][j+1] = j+1;
                }else{
                    lcss[i+1][j+1] = lcss[i+1][j];
                    lastX[i+1][j+1] = i+1;
                    lastY[i+1][j+1] = j;
                }
            }
        }

        int max = -1;
        for(int i = 1; i < lcss[lcss.length-1].length; i++){
            if(lcss[lcss.length-1][i] > max){
                max = lcss[lcss.length-1][i];
            }
        }
        return 1-((double)max/m);
    }

    public static double stdv_s(double[] input){

        double sumx = 0;
        double sumx2 = 0;


        for(int j = 0; j < input.length; j++){
            sumx+=input[j];
            sumx2+=input[j]*input[j];
        }

        int n = input.length;
        double mean = sumx/n;
        double standardDev = Math.sqrt(sumx2/(n-1) - mean*mean);

        return standardDev;

    }
    
    public static double stdv_p(Instances input){

        double sumx = 0;
        double sumx2 = 0;
        double[] ins2array;
        for(int i = 0; i < input.numInstances(); i++){
            ins2array = input.instance(i).toDoubleArray();
            for(int j = 0; j < ins2array.length-1; j++){//-1 to avoid classVal
                sumx+=ins2array[j];
                sumx2+=ins2array[j]*ins2array[j];
            }
        }
        int n = input.numInstances()*(input.numAttributes()-1);
        double mean = sumx/n;
        return Math.sqrt(sumx2/(n)-mean*mean);

    }

    public static int[] getInclusive10(int min, int max){
        int[] output = new int[10];

        double diff = (double)(max-min)/9;
        double[] doubleOut = new double[10];
        doubleOut[0] = min;
        output[0] = min;
        for(int i = 1; i < 9; i++){
            doubleOut[i] = doubleOut[i-1]+diff;
            output[i] = (int)Math.round(doubleOut[i]);
        }
        output[9] = max; // to make sure max isn't omitted due to double imprecision
        return output;
    }

    public static double[] getInclusive10(double min, double max){
        double[] output = new double[10];
        double diff = (double)(max-min)/9;
        output[0] = min;
        for(int i = 1; i < 9; i++){
            output[i] = output[i-1]+diff;
        }
        output[9] = max;
        
        return output;
    }


}
