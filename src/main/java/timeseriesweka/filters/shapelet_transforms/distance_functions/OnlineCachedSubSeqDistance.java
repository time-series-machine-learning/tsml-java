/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.distance_functions;

    /*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
*/

import weka.core.Instances;
import timeseriesweka.filters.shapelet_transforms.Shapelet;

/**
 *
 * @author raj09hxu
 */
public class OnlineCachedSubSeqDistance extends SubSeqDistance{


    protected Stats stats;
    protected double[][] data;
    
    
    @Override
    public void init(Instances dataInst)
    {
        stats = new Stats();
        
        //Normalise all time series for further processing
        int dataSize = dataInst.numInstances();
        
        data = new double[dataSize][];
        for (int i = 0; i < dataSize; i++)
        {
            data[i] = zNormalise(dataInst.get(i).toDoubleArray(), true);
        }
    }
    
    @Override
    public void setShapelet(Shapelet shp)
    {
        super.setShapelet(shp);
        
        //for transforming we don't want to use the stats. it doesn't make sense.
        stats = null;
    }
    
    @Override
    public void setSeries(int seriesId) {
        super.setSeries(seriesId);
        
        //do some extra stats.
        stats.computeStats(seriesId, data);
    }
    
    @Override
    public double calculate(double[] timeSeries, int timeSeriesId) {
        //if the stats object is null, use normal calculations.
        if(stats == null)
            return super.calculate(timeSeries,timeSeriesId);
                
        //the series we're comparing too.
        stats.setCurrentY(timeSeriesId, data);
        
        double minSum = Double.MAX_VALUE;
        int subLength = length;
        

        double xMean = stats.getMeanX(startPos, subLength);
        double xStdDev = stats.getStdDevX(startPos, subLength);
        
        double yMean;
        double yStdDev;
        double crossProd;

        // Scan through all possible subsequences of two
        for (int v = 0; v < timeSeries.length - subLength; v++)
        {
            yMean = stats.getMeanY(v, subLength);
            yStdDev = stats.getStdDevY(v, subLength);
            crossProd = stats.getSumOfProds(startPos, v, subLength);

            double cXY = 0.0;
            if (xStdDev != 0 && yStdDev != 0)
            {
                cXY = (crossProd - (subLength * xMean * yMean)) / ((double)subLength * xStdDev * yStdDev);
            }

            double dist = 2.0 * (1.0 - cXY);

            if (dist < minSum)
            {
                minSum = dist;
            }
        }

        return minSum;
    }
    
    
    /**
     * A class for holding relevant statistics for any given candidate series
     * and all time series TO DO: CONVERT IT ALL TO FLOATS
     * Aaron: Changed to floats, why?
     */
    public static class Stats
    {

        private double[] cummSumsX;
        private double[] cummSqSumsX;
        
        private double[]cummSumsY;
        private double[]cummSqSumsY;
        
        private double[][] crossProdsY;
        private int xIndex;

        /**
         * Default constructor
         */
        public Stats()
        {
            cummSumsX = null;
            cummSqSumsX = null;
            cummSumsY = null;
            cummSqSumsY = null;
            crossProdsY = null;
            xIndex = -1;
        }


        /**
         * A method to set current time series that is being examined.
         *
         * @param yIndex time series index
         */
        public void setCurrentY(int yIndex, double[][] data)
        {
            //calculate the cumulative sums for the new series.
            double[][] sums = computeCummSums(data[yIndex]);
            cummSumsY = sums[0];
            cummSqSumsY = sums[1];

            //Compute cross products between candidate series and current series
            crossProdsY = computeCrossProd(data[xIndex], data[yIndex]);
            
        }

        /**
         * A method to retrieve the mean value of a whole candidate sub-series.
         *
         * @param startPos start position of the candidate
         * @param subLength length of the candidate
         * @return mean value of sub-series
         */
        public double getMeanX(int startPos, int subLength)
        {
            double diff = cummSumsX[startPos + subLength] - cummSumsY[startPos];
            return diff / (double) subLength;
        }

        /**
         * A method to retrieve the mean value for time series sub-series. Note
         * that the current Y must be set prior invoking this method.
         *
         * @param startPos start position of the sub-series
         * @param subLength length of the sub-series
         * @return mean value of sub-series
         */
        public double getMeanY(int startPos, int subLength)
        {
            double diff = cummSumsY[startPos + subLength] - cummSumsY[startPos];
            return diff / (double) subLength;
        }

        /**
         * A method to retrieve the standard deviation of a whole candidate
         * sub-series.
         *
         * @param startPos start position of the candidate
         * @param subLength length of the candidate
         * @return standard deviation of the candidate sub-series
         */
        public double getStdDevX(int startPos, int subLength)
        {
            double diff = cummSqSumsX[startPos + subLength] - cummSqSumsX[startPos];
            double meanSqrd = getMeanX(startPos, subLength) * getMeanX(startPos, subLength);
            double temp = diff / (double) subLength;
            double temp1 = temp - meanSqrd;
            return Math.sqrt(temp1);
        }

        /**
         * A method to retrieve the standard deviation for time series
         * sub-series. Note that the current Y must be set prior invoking this
         * method.
         *
         * @param startPos start position of the sub-series
         * @param subLength length of the sub-series
         * @return standard deviation of sub-series
         */
        public double getStdDevY(int startPos, int subLength)
        {
            double diff = cummSqSumsY[startPos + subLength] - cummSqSumsY[startPos];
            double meanSqrd = getMeanX(startPos, subLength) * getMeanX(startPos, subLength);
            double temp = diff / (double) subLength;
            double temp1 = temp - meanSqrd;
            return Math.sqrt(temp1);
        }

        /**
         * A method to retrieve the cross product of whole candidate sub-series
         * and time series sub-series. Note that the current Y must be set prior
         * invoking this method.
         *
         * @param startX start position of the whole candidate sub-series
         * @param startY start position of the time series sub-series
         * @param length length of the both sub-series
         * @return sum of products for a given overlap between two sub=series
         */
        public double getSumOfProds(int startX, int startY, int length)
        {
            return crossProdsY[startX + length][startY + length] - crossProdsY[startX][startY];
        }

        private double[][] computeCummSums(double[] currentSeries)
        {

            double[][] output = new double[2][];
            output[0] = new double[currentSeries.length];
            output[1] = new double[currentSeries.length];
            output[0][0] = 0;
            output[1][0] = 0;

            //Compute stats for a given series instance
            for (int i = 1; i < currentSeries.length; i++)
            {
                output[0][i] = (double) (output[0][i - 1] + currentSeries[i - 1]);                         //Sum of vals
                output[1][i] = (double) (output[1][i - 1] + (currentSeries[i - 1] * currentSeries[i - 1]));  //Sum of squared vals
            }

            return output;
        }

        private double[][] computeCrossProd(double[] x, double[] y)
        {
            //im assuming the starting from 1 with -1 is because of class values at the end.
            
            double[][] output = new double[x.length][y.length];

            for (int u = 1; u < x.length; u++)
            {
                for (int v = 1; v < y.length; v++)
                {
                    int t;  //abs(u-v)
                    if (v < u)
                    {
                        t = u - v;
                        output[u][v] = (double) (output[u - 1][v - 1] + (x[v - 1 + t] * y[v - 1]));
                    }
                    else
                    {//else v >= u
                        t = v - u;
                        output[u][v] = (double) (output[u - 1][v - 1] + (x[u - 1] * y[u - 1 + t]));
                    }
                }
            }

            return output;
        }

        /**
         * A method to compute statistics for a given candidate series index and
         * normalised time series
         *
         * @param candidateInstIndex index of the candidate within the time
         * series database
         * @param data the normalised database of time series
         */
        public void computeStats(int candidateInstIndex, double[][] data)
        {
            xIndex = candidateInstIndex;

            double[][] sums = computeCummSums(data[xIndex]);
            cummSumsX = sums[0];
            cummSqSumsX = sums[1];
        }
    }
    
    
}
