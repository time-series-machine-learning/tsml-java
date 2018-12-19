/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.quality_measures;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import utilities.class_distributions.ClassDistribution;
import timeseriesweka.filters.shapelet_transforms.OrderLineObj;

/**
 *
 * @author raj09hxu
 */
/**
     * A class for calculating the Mood's Median statistic of a shapelet,
     * according to the set of distances from the shapelet to a dataset.
     */
    public class MoodsMedian implements ShapeletQualityMeasure, Serializable
    {

        protected MoodsMedian(){}
        
        /**
         * A method to calculate the quality of a FullShapeletTransform, given
         * the orderline produced by computing the distance from the shapelet to
         * each element of the dataset.
         *
         * @param orderline the pre-computed set of distances for a dataset to a
         * single shapelet
         * @param classDistributions the distibution of all possible class
         * values in the orderline
         * @return a measure of shapelet quality according to Mood's Median
         */
        @Override
        public double calculateQuality(List<OrderLineObj> orderline, ClassDistribution classDistributions)
        {

            //naive implementation as a benchmark for finding median - actually faster than manual quickSelect! Probably due to optimised java implementation
            Collections.sort(orderline);
            int lengthOfOrderline = orderline.size();
            double median;
            if (lengthOfOrderline % 2 == 0)
            {
                median = (orderline.get(lengthOfOrderline / 2 - 1).getDistance() + orderline.get(lengthOfOrderline / 2).getDistance()) / 2;
            }
            else
            {
                median = orderline.get(lengthOfOrderline / 2).getDistance();
            }

            int totalCount = orderline.size();
            int countBelow = 0;
            int countAbove = 0;
            int numClasses = classDistributions.size();
            int[] classCountsBelowMedian = new int[numClasses];
            int[] classCountsAboveMedian = new int[numClasses];

            double distance;
            double classVal;
            int countSoFar;
            for (OrderLineObj orderline1 : orderline)
            {
                distance = orderline1.getDistance();
                classVal = orderline1.getClassVal();
                if (distance < median)
                {
                    countBelow++;
                    classCountsBelowMedian[(int) classVal]++;
                }
                else
                {
                    countAbove++;
                    classCountsAboveMedian[(int) classVal]++;
                }
            }

            double chi = 0;
            double expectedAbove, expectedBelow;
            for (int i = 0; i < numClasses; i++)
            {
                expectedBelow = (double) (countBelow * classDistributions.get((double) i)) / totalCount;
                chi += ((classCountsBelowMedian[i] - expectedBelow) * (classCountsBelowMedian[i] - expectedBelow)) / expectedBelow;

                expectedAbove = (double) (countAbove * classDistributions.get((double) i)) / totalCount;
                chi += ((classCountsAboveMedian[i] - expectedAbove)) * (classCountsAboveMedian[i] - expectedAbove) / expectedAbove;
            }

            if (Double.isNaN(chi))
            {
                chi = 0; // fix for cases where the shapelet is a straight line and chi is calc'd as NaN
            }
            return chi;
        }

        @Override
        public double calculateSeperationGap(List<OrderLineObj> orderline) {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

    }
