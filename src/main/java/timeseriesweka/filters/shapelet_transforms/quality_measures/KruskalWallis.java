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
     * A class for calculating the Kruskal-Wallis statistic of a shapelet,
     * according to the set of distances from the shapelet to a dataset.
     */
    public class KruskalWallis implements ShapeletQualityMeasure, Serializable
    {

        protected KruskalWallis(){}
        
        /**
         * A method to calculate the quality of a FullShapeletTransform, given
         * the orderline produced by computing the distance from the shapelet to
         * each element of the dataset.
         *
         * @param orderline the pre-computed set of distances for a dataset to a
         * single shapelet
         * @param classDistribution the distibution of all possible class values
         * in the orderline
         * @return a measure of shapelet quality according to Kruskal-Wallis
         */
        @Override
        public double calculateQuality(List<OrderLineObj> orderline, ClassDistribution classDistribution)
        {
            // sort
            Collections.sort(orderline);
            int numClasses = classDistribution.size();
            int[] classRankCounts = new int[numClasses];
            double[] classRankMeans = new double[numClasses];

            double lastDistance = orderline.get(0).getDistance();
            double thisDistance = lastDistance;
            double classVal = orderline.get(0).getClassVal();
            classRankCounts[(int) classVal] += 1;

            int duplicateCount = 0;

            for (int i = 1; i < orderline.size(); i++)
            {
                thisDistance = orderline.get(i).getDistance();
                if (duplicateCount == 0 && thisDistance != lastDistance)
                { // standard entry
                    classRankCounts[(int) orderline.get(i).getClassVal()] += i + 1;

                }
                else if (duplicateCount > 0 && thisDistance != lastDistance)
                { // non-duplicate following duplicates
                    // set ranks for dupicates

                    double minRank = i - duplicateCount;
                    double maxRank = i;
                    double avgRank = (minRank + maxRank) / 2;

                    for (int j = i - duplicateCount - 1; j < i; j++)
                    {
                        classRankCounts[(int) orderline.get(j).getClassVal()] += avgRank;
                    }

                    duplicateCount = 0;
                    // then set this rank
                    classRankCounts[(int) orderline.get(i).getClassVal()] += i + 1;
                }
                else
                {// thisDistance==lastDistance
                    if (i == orderline.size() - 1)
                    { // last one so must do the avg ranks here (basically copied from above, BUT includes this element too now)

                        double minRank = i - duplicateCount;
                        double maxRank = i + 1;
                        double avgRank = (minRank + maxRank) / 2;

                        for (int j = i - duplicateCount - 1; j <= i; j++)
                        {
                            classRankCounts[(int) orderline.get(j).getClassVal()] += avgRank;
                        }
                    }
                    duplicateCount++;
                }
                lastDistance = thisDistance;
            }

            //3) overall mean rank
            double overallMeanRank = (1.0 + orderline.size()) / 2;

            //4) sum of squared deviations from the overall mean rank
            double s = 0;
            for (int i = 0; i < numClasses; i++)
            {
                classRankMeans[i] = (double) classRankCounts[i] / classDistribution.get((double) i);
                s += classDistribution.get((double) i) * (classRankMeans[i] - overallMeanRank) * (classRankMeans[i] - overallMeanRank);
            }

            //5) weight s with the scale factor
            double h = 12.0 / (orderline.size() * (orderline.size() + 1)) * s;

            return h;
        }

        @Override
        public double calculateSeperationGap(List<OrderLineObj> orderline) {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }
    }
