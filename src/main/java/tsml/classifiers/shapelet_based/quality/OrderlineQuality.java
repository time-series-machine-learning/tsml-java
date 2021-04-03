package tsml.classifiers.shapelet_based.quality;

import tsml.classifiers.shapelet_based.distances.ShapeletDistanceFunction;
import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;
import tsml.transformers.shapelet_tools.OrderLineObj;
import utilities.class_counts.ClassCounts;
import utilities.class_counts.TreeSetClassCounts;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

public class OrderlineQuality extends ShapeletQualityFunction {

    public OrderlineQuality(TimeSeriesInstances instances,
                            ShapeletDistanceFunction distance){
        super(instances,distance);
    }

    @Override
    public double calculate(ShapeletMV candidate) {



        // create orderline by looping through data set and calculating the subsequence
        // distance from candidate to all data, inserting in order.
        ArrayList<OrderLineObj> orderline = new ArrayList<>();

        int dataSize = trainInstances.numInstances();

        for (int i = 0; i < dataSize; i++) {



            double d = distance.calculate(candidate,trainInstances.get(i));

            //this could be binarised or normal.
            double classVal = classIndexes[i];

            // without early abandon, it is faster to just add and sort at the end
            orderline.add(new OrderLineObj(d, classVal));


        }


        return calculateQuality(orderline,classCounts);
    }

    public double calculateQuality(List<OrderLineObj> orderline, int[] classDistribution)
    {
        Collections.sort(orderline);
        // for each split point, starting between 0 and 1, ending between end-1 and end
        // addition: track the last threshold that was used, don't bother if it's the same as the last one
        double lastDist = -1;//orderline.get(0).getDistance(); // must be initialised as not visited(no point breaking before any data!)
        double thisDist = -1;

        double bsfGain = -1;
        double threshold = -1;

        // initialise class counts
        ClassCounts lessClasses = new TreeSetClassCounts();
        ClassCounts greaterClasses = new TreeSetClassCounts();

        // parent entropy will always be the same, so calculate just once
        double parentEntropy = entropy(classDistribution);

        int sumOfAllClasses = 0;
        for (int j=0;j<classDistribution.length;j++)
        {
            lessClasses.put(j, 0);
            greaterClasses.put(j, classDistribution[j]);
            sumOfAllClasses += classDistribution[j];
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
                    threshold = (thisDist - lastDist) / 2 + lastDist;
                }
            }
            lastDist = thisDist;
        }
        return bsfGain;
    }

    public static double calculateSplitThreshold(List<OrderLineObj> orderline, int[] classDistribution){
        Collections.sort(orderline);
        // for each split point, starting between 0 and 1, ending between end-1 and end
        // addition: track the last threshold that was used, don't bother if it's the same as the last one
        double lastDist = orderline.get(0).getDistance(); // must be initialised as not visited(no point breaking before any data!)
        double thisDist = 0;

        double bsfGain = -1;
        double threshold = 1;

        // initialise class counts
        ClassCounts lessClasses = new TreeSetClassCounts();
        ClassCounts greaterClasses = new TreeSetClassCounts();

        // parent entropy will always be the same, so calculate just once
        double parentEntropy = entropy(classDistribution);

        int sumOfAllClasses = 0;
        for (int j=0;j<classDistribution.length;j++)
        {
            lessClasses.put(j, 0);
            greaterClasses.put(j, classDistribution[j]);
            sumOfAllClasses += classDistribution[j];
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
                    threshold = (thisDist - lastDist) / 2 + lastDist;
                }
            }
            lastDist = thisDist;
        }
        return threshold;
    }

    public static double entropy(int[] classDistributions)
    {
        if (classDistributions.length == 1)
        {
            return 0;
        }

        double thisPart;
        double toAdd;
        int total = 0;
        //Aaron: should be simpler than iterating using the keySet.
        //Values is backed by the Map so it doesn't need to be constructed.

        for (int j=0;j<classDistributions.length;j++) {
            total += classDistributions[j];
        }

        // to avoid NaN calculations, the individual parts of the entropy are calculated and summed.
        // i.e. if there is 0 of a class, then that part would calculate as NaN, but this can be caught and
        // set to 0.
        //Aaron:  Instead of using the keyset to loop through, use the underlying Array to iterate through, ordering of calculations doesnt matter.
        //just that we do them all. so i think previously it was n log n, now should be just n.
        double entropy = 0;
        for (int j=0;j<classDistributions.length;j++)
        {
            thisPart = (double) classDistributions[j] / total;
            toAdd = -thisPart * Math.log10(thisPart) / Math.log10(2);
            //Aaron: if its not NaN we can add it, if it was NaN we'd just add 0.
            if (!Double.isNaN(toAdd))
            {
                entropy += toAdd;
            }
        }

        return entropy;
    }

    public double calculateSeperationGap(List<OrderLineObj> orderline) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public static double entropy(ClassCounts classDistributions)
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
}
