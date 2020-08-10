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
package tsml.transformers;

//import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;

import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.transformers.shapelet_tools.OrderLineObj;
import utilities.class_counts.TreeSetClassCounts;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;

/**
 * A binary filter that uses information gain quality measure to determine the
 * split point/ copyright: Anthony Bagnall
 * 
 * @author Jon Hills j.hills@uea.ac.uk
 */
public class BinaryTransform implements TrainableTransformer {
    private boolean isFit = false;
    private double[] splits;


    @Override
    public Instances determineOutputFormat(Instances inputFormat) {
        int length = inputFormat.numAttributes();
        if (inputFormat.classIndex() >= 0)
            length--;

        // Set up instances size and format.
        ArrayList<Attribute> atts = new ArrayList<>();
        ArrayList<String> attributeValues = new ArrayList<>();
        attributeValues.add("0");
        attributeValues.add("1");

        String name;
        for (int i = 0; i < length; i++) {
            name = "Binary_" + i;
            atts.add(new Attribute(name, attributeValues));
        }
        if (inputFormat.classIndex() >= 0) { // Classification set, set class
            // Get the class values as a fast vector
            Attribute target = inputFormat.attribute(inputFormat.classIndex());

            ArrayList<String> vals = new ArrayList<>();
            for (int i = 0; i < target.numValues(); i++)
                vals.add(target.value(i));
            atts.add(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
        }
        Instances result = new Instances("Binary" + inputFormat.relationName(), atts, inputFormat.numInstances());
        if (inputFormat.classIndex() >= 0) {
            result.setClassIndex(result.numAttributes() - 1);
        }
        return result;

    }

    public double findSplitValue(Instances data, double[] vals, double[] classes) {
        // return 1;
        // Put into an order list
        ArrayList<OrderLineObj> list = new ArrayList<OrderLineObj>();
        for (int i = 0; i < vals.length; i++)
            list.add(new OrderLineObj(vals[i], classes[i]));
        // Sort the vals
        int[] tree = new TreeSetClassCounts(data).values().stream().mapToInt(e -> e.intValue()).toArray();
        Collections.sort(list);
        return infoGainThreshold(list, tree);
    }

    private static double entropy(int[] classDistributions) {
        if (classDistributions.length == 1) {
            return 0;
        }

        double thisPart;
        double toAdd;
        int total = 0;
        for (int i : classDistributions) {
            total += i;
        }

        // to avoid NaN calculations, the individual parts of the entropy are calculated
        // and summed.
        // i.e. if there is 0 of a class, then that part would calculate as NaN, but
        // this can be caught and
        // set to 0.
        ArrayList<Double> entropyParts = new ArrayList<Double>();
        for (int i : classDistributions)  {
            thisPart = (double) i / total;
            toAdd = -thisPart * Math.log10(thisPart) / Math.log10(2);
            if (Double.isNaN(toAdd))
                toAdd = 0;
            entropyParts.add(toAdd);
        }

        double entropy = 0;
        for (int i = 0; i < entropyParts.size(); i++) {
            entropy += entropyParts.get(i);
        }
        return entropy;
    }

    public static double infoGainThreshold(ArrayList<OrderLineObj> orderline, int[] classDistribution) {
        // for each split point, starting between 0 and 1, ending between end-1 and end
        // addition: track the last threshold that was used, don't bother if it's the
        // same as the last one
        double lastDist = orderline.get(0).getDistance(); // must be initialised as not visited(no point breaking before
                                                          // any data!)
        double thisDist = -1;

        double bsfGain = -1;
        double threshold = -1;

        // check that there is actually a split point
        // for example, if all

        for (int i = 1; i < orderline.size(); i++) {
            thisDist = orderline.get(i).getDistance();
            if (i == 1 || thisDist != lastDist) { // check that threshold has moved(no point in sampling identical
                                                  // thresholds)- special case - if 0 and 1 are the same dist

                // count class instances below and above threshold
                int[] lessClasses = new int[classDistribution.length];
                int[] greaterClasses = new int[classDistribution.length];

                int sumOfLessClasses = 0;
                int sumOfGreaterClasses = 0;

                // visit those below threshold
                for (int j = 0; j < i; j++) {
                    lessClasses[(int)orderline.get(j).getClassVal()]++;
                    sumOfLessClasses++;
                }

                for (int j = i; j < orderline.size(); j++) {
                    greaterClasses[(int)orderline.get(j).getClassVal()]++;
                    sumOfGreaterClasses++;
                }


                int sumOfAllClasses = sumOfLessClasses + sumOfGreaterClasses;

                double parentEntropy = entropy(classDistribution);

                // calculate the info gain below the threshold
                double lessFrac = (double) sumOfLessClasses / sumOfAllClasses;
                double entropyLess = entropy(lessClasses);
                // calculate the info gain above the threshold
                double greaterFrac = (double) sumOfGreaterClasses / sumOfAllClasses;
                double entropyGreater = entropy(greaterClasses);

                double gain = parentEntropy - lessFrac * entropyLess - greaterFrac * entropyGreater;
                // System.out.println(parentEntropy+" - "+lessFrac+" * "+entropyLess+" -
                // "+greaterFrac+" * "+entropyGreater);
                // System.out.println("gain calc:"+gain);
                if (gain > bsfGain) {
                    bsfGain = gain;
                    threshold = (thisDist - lastDist) / 2 + lastDist;
                }
            }
            lastDist = thisDist;
        }
        return threshold;
    }

    public double findSplitValue(TimeSeriesInstances data, double[] vals, double[] classes) {
        // return 1;
        // Put into an order list
        ArrayList<OrderLineObj> list = new ArrayList<OrderLineObj>();
        for (int i = 0; i < vals.length; i++)
            list.add(new OrderLineObj(vals[i], classes[i]));
        // Sort the vals
        int[] tree = data.getClassCounts();
        Collections.sort(list);
        return infoGainThreshold(list, tree);
    }


    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        return null;
    }

    @Override
    public void fit(TimeSeriesInstances data) {
        
        /*
        splits = new double[data.getMaxLength()];
        
        int[] classes = new int[data.numInstances()];
        int i=0;
        for(TimeSeriesInstance inst :data)
            classes[i++] = inst.getLabelIndex();


        for(TimeSeriesInstance inst :data)
            // Get values of attribute j
            for(TimeSeries ts : inst){
                // find the IG split point
                splits[j] = findSplitValue(data, vals, classes);
            }
           
            
        }

        isFit = true;*/
    }

    @Override
    public Instance transform(Instance inst) {
        Instance newInst = new DenseInstance(inst.numAttributes());
        for (int j = 0; j < inst.numAttributes(); j++) {
            if (j != inst.classIndex()) {
                if (inst.value(j) < splits[j])
                    newInst.setValue(j, 0);
                else
                    newInst.setValue(j, 1);
            } else
                newInst.setValue(j, inst.classValue());
        }
        return newInst;
    }

    @Override
    public void fit(Instances data) {

        splits = new double[data.numAttributes()];
        double[] classes = new double[data.numInstances()];
        for (int i = 0; i < classes.length; i++)
            classes[i] = data.instance(i).classValue();
        for (int j = 0; j < data.numAttributes(); j++) { // for each data
            if (j != data.classIndex()) {

                // Get values of attribute j
                double[] vals = new double[data.numInstances()];
                for (int i = 0; i < data.numInstances(); i++)
                    vals[i] = data.instance(i).value(j);
                // find the IG split point
                splits[j] = findSplitValue(data, vals, classes);
            }
        }

        isFit = true;
        
    }

    @Override
    public boolean isFit() {
        return isFit;
    }


    
       
}
