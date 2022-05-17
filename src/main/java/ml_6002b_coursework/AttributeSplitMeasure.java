package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Interface for alternative attribute split measures for Part 2.2 of the coursework
 */
public abstract class AttributeSplitMeasure {

    public abstract double computeAttributeQuality(Instances data, Attribute att) throws Exception;

    /**
     * Splits a dataset according to the values of a nominal attribute.
     *
     * @param data the data which is to be split
     * @param att the attribute to be used for splitting
     * @return the sets of instances produced by the split
     */
    public Instances[] splitData(Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];
        for (int i = 0; i < att.numValues(); i++) {
            splitData[i] = new Instances(data, data.numInstances());
        }

        for (Instance inst: data) {
            splitData[(int) inst.value(att)].add(inst);
        }

        for (Instances split : splitData) {
            split.compactify();
        }


        return splitData;
    }

    /**
     * Calculates the mean value of data
     * @param data
     * @param att
     * @return mean
     */
//    public static double splitValue(Instances data, Attribute att) {
////        if (numArray.length % 2 == 0)
////            median = ((double)numArray[numArray.length/2] + (double)numArray[numArray.length/2 - 1])/2;
////        else
////            median = (double) numArray[numArray.length/2];
////        meanValue = data.meanOrMode(att);
//        System.out.println(att);
//        return data.meanOrMode(att);
//    }

    /**
     * Splits data with a numeric attribute
     * @param data
     * @param att
     * @return Nominal array
     */
    public Instances[]splitDataOnNumeric(Instances data, Attribute att) {
        //calculates average value
        double meanValue = data.meanOrMode(att);

        Instances[] splitDataNumeric = new Instances[2];
        splitDataNumeric[0] = new Instances(data, data.numInstances());
        splitDataNumeric[1] = new Instances(data, data.numInstances());

        for (Instance inst: data) {
            splitDataNumeric[inst.value(att) < meanValue ? 0 : 1].add(inst);
        }
//        System.out.println(splitDataNumeric[0]);
//        System.out.println();
//        System.out.println(splitDataNumeric[1]);
//        System.out.println();
//        System.out.println(meanValue+" mean");
//        System.out.println();

        for (Instances split : splitDataNumeric) {
            split.compactify();
        }

        return splitDataNumeric;
    }
}
