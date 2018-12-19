/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.quality_measures;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import utilities.class_distributions.ClassDistribution;
import timeseriesweka.filters.shapelet_transforms.OrderLineObj;
/**
 *
 * @author raj09hxu
 */
/**
     * A class for calculating the F-Statistic of a shapelet, according to the
     * set of distances from the shapelet to a dataset.
     */
    public class FStat implements ShapeletQualityMeasure, Serializable
    {

        protected FStat(){
            
        }
        
        /**
         * A method to calculate the quality of a FullShapeletTransform, given
         * the orderline produced by computing the distance from the shapelet to
         * each element of the dataset.
         *
         * @param orderline the pre-computed set of distances for a dataset to a
         * single shapelet
         * @param classDistribution the distibution of all possible class values
         * in the orderline
         * @return a measure of shapelet quality according to f-stat
         */
        @Override
        public double calculateQuality(List<OrderLineObj> orderline, ClassDistribution classDistribution)
        {
            Collections.sort(orderline);
            int numClasses = classDistribution.size();
            int numInstances = orderline.size();

            double[] sums = new double[numClasses];
            double[] sumsSquared = new double[numClasses];
            double[] sumOfSquares = new double[numClasses];

            for (int i = 0; i < numClasses; i++)
            {
                sums[i] = 0;
                sumsSquared[i] = 0;
                sumOfSquares[i] = 0;
            }

            for (OrderLineObj orderline1 : orderline)
            {
                int c = (int) orderline1.getClassVal();
                double thisDist = orderline1.getDistance();
                sums[c] += thisDist;
                sumOfSquares[c] += thisDist * thisDist;
            }

            for (int i = 0; i < numClasses; i++)
            {
                sumsSquared[i] = sums[i] * sums[i];
            }

            double ssTotal = 0;
            double part1 = 0;
            double part2 = 0;

            for (int i = 0; i < numClasses; i++)
            {
                part1 += sumOfSquares[i];
                part2 += sums[i];
            }

            part2 *= part2;
            part2 /= numInstances;
            ssTotal = part1 - part2;

            double ssAmoung = 0;
            part1 = 0;
            part2 = 0;
            for (int i = 0; i < numClasses; i++)
            {
                part1 += (double) sumsSquared[i] / classDistribution.get((double) i);//.data[i].size();
                part2 += sums[i];
            }
            ssAmoung = part1 - (part2 * part2) / numInstances;
            double ssWithin = ssTotal - ssAmoung;

            int dfAmoung = numClasses - 1;
            int dfWithin = numInstances - numClasses;

            double msAmoung = ssAmoung / dfAmoung;
            double msWithin = ssWithin / dfWithin;

            double f = msAmoung / msWithin;

            return Double.isNaN(f) ? 0.0 : f;
        }

        /**
         *
         * @param orderline
         * @param classDistribution
         * @return a va
         */
        public double calculateQualityNew(List<OrderLineObj> orderline, Map<Double, Integer> classDistribution)
        {
            Collections.sort(orderline);
            int numClasses = classDistribution.size();
            int numInstances = orderline.size();

            double[] sums = new double[numClasses];
            double[] sumsSquared = new double[numClasses];
            double[] sumOfSquares = new double[numClasses];

            for (int i = 0; i < orderline.size(); i++)
            {
                int c = (int) orderline.get(i).getClassVal();
                double thisDist = orderline.get(i).getDistance();
                sums[c] += thisDist;
                sumOfSquares[c] += thisDist * thisDist;
            }

            double ssTotal = 0;
            double part1 = 0;
            double part2 = 0;

            for (int i = 0; i < numClasses; i++)
            {
                sumsSquared[i] = sums[i] * sums[i];
                part1 += sumOfSquares[i];
                part2 += sums[i];
            }

            part2 *= part2;
            part2 /= numInstances;
            ssTotal = part1 - part2;

            double ssAmoung = 0;
            part1 = 0;
            part2 = 0;
            for (int i = 0; i < numClasses; i++)
            {
                part1 += (double) sumsSquared[i] / classDistribution.get((double) i);//.data[i].size();
                part2 += sums[i];
            }
            ssAmoung = part1 - (part2 * part2) / numInstances;
            double ssWithin = ssTotal - ssAmoung;

            int dfAmoung = numClasses - 1;
            int dfWithin = numInstances - numClasses;

            double msAmoung = ssAmoung / dfAmoung;
            double msWithin = ssWithin / dfWithin;

            double f = msAmoung / msWithin;

            return f;
        }

        @Override
        public double calculateSeperationGap(List<OrderLineObj> orderline) {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }
    }
