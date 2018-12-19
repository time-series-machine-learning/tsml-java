/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.quality_measures;

import java.io.Serializable;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import utilities.class_distributions.ClassDistribution;
import utilities.class_distributions.TreeSetClassDistribution;
import timeseriesweka.filters.shapelet_transforms.OrderLineObj;
/**
 *
 * @author raj09hxu
 */

    /**
     * A class for calculating the information gain of a shapelet, according to
     * the set of distances from the shapelet to a dataset.
     */
    public class InformationGain implements ShapeletQualityMeasure, Serializable 
    {

        protected InformationGain(){
            
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
         * @return a measure of shapelet quality according to information gain
         */
        @Override
        public double calculateQuality(List<OrderLineObj> orderline, ClassDistribution classDistribution)
        {
            Collections.sort(orderline);
            // for each split point, starting between 0 and 1, ending between end-1 and end
            // addition: track the last threshold that was used, don't bother if it's the same as the last one
            double lastDist = -1;//orderline.get(0).getDistance(); // must be initialised as not visited(no point breaking before any data!)
            double thisDist = -1;

            double bsfGain = -1;

            // initialise class counts
            ClassDistribution lessClasses = new TreeSetClassDistribution();
            ClassDistribution greaterClasses = new TreeSetClassDistribution();

            // parent entropy will always be the same, so calculate just once
            double parentEntropy = entropy(classDistribution);

            int sumOfAllClasses = 0;
            for (double j : classDistribution.keySet())
            {
                lessClasses.put(j, 0);
                greaterClasses.put(j, classDistribution.get(j));
                sumOfAllClasses += classDistribution.get(j);
            }
            int sumOfLessClasses = 0;
            int sumOfGreaterClasses = sumOfAllClasses;

            double thisClassVal;
            int oldCount;

            for (OrderLineObj ol : orderline)
            {
                thisDist = ol.getDistance();

                //move the threshold along one (effectively by adding this dist to lessClasses
                thisClassVal = ol.getClassVal();
                oldCount = lessClasses.get(thisClassVal) + 1;
                lessClasses.put(thisClassVal, oldCount);
                oldCount = greaterClasses.get(thisClassVal) - 1;
                greaterClasses.put(thisClassVal, oldCount);

                // adjust counts - maybe makes more sense if these are called counts, rather than sums!
                sumOfLessClasses++;
                sumOfGreaterClasses--;

                // check to see if the threshold has moved (ie if thisDist isn't the same as lastDist)
                // important, else gain calculations will be made 'in the middle' of a threshold, resulting in different info gain for
                // the split point, that won't actually be valid as it is 'on' a distances, rather than 'between' them/
                if (thisDist != lastDist)
                {

                    // calculate the info gain below the threshold
                    double lessFrac = (double) sumOfLessClasses / sumOfAllClasses;
                    double entropyLess = entropy(lessClasses);

                    // calculate the info gain above the threshold
                    double greaterFrac = (double) sumOfGreaterClasses / sumOfAllClasses;
                    double entropyGreater = entropy(greaterClasses);

                    double gain = parentEntropy - lessFrac * entropyLess - greaterFrac * entropyGreater;
                    if (gain > bsfGain)
                    {
                        bsfGain = gain;
                    }
                }
                lastDist = thisDist;
            }
            return bsfGain;
        }

        public static double entropy(ClassDistribution classDistributions)
        {
            if (classDistributions.size() == 1)
            {
                return 0;
            }

            double thisPart;
            double toAdd;
            int total = 0;
            //Aaron: should be simpler than iterating using the keySet.
            //Values is backed by the Map so it doesn't need to be constructed.
            Collection<Integer> values = classDistributions.values();
            for (Integer d : values)
            {
                total += d;
            }

            // to avoid NaN calculations, the individual parts of the entropy are calculated and summed.
            // i.e. if there is 0 of a class, then that part would calculate as NaN, but this can be caught and
            // set to 0. 
            //Aaron:  Instead of using the keyset to loop through, use the underlying Array to iterate through, ordering of calculations doesnt matter.
            //just that we do them all. so i think previously it was n log n, now should be just n.
            double entropy = 0;
            for (Integer d : values)
            {
                thisPart = (double) d / total;
                toAdd = -thisPart * Math.log10(thisPart) / Math.log10(2);
                //Aaron: if its not NaN we can add it, if it was NaN we'd just add 0.
                if (!Double.isNaN(toAdd))
                {
                    entropy += toAdd;
                }
            }

            return entropy;
        }

        @Override
        public double calculateSeperationGap(List<OrderLineObj> orderline) {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

    }
