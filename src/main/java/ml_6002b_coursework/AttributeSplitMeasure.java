package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;

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

    public Instances[]splitDataNumeric(Instances data, Attribute att) {
        //calculates average value
        double meanValue = 0.0;
        for(Instance instance : data) {
            meanValue +=instance.value(att);
        }
        meanValue = meanValue / data.size();

        Instances[] splitDataNumeric = new Instances[2];
        splitDataNumeric[0] = new Instances(data, data.numInstances());
        splitDataNumeric[1] = new Instances(data, data.numInstances());

        for (Instance inst: data) {
            splitDataNumeric[(int) inst.value(att)].add(inst);
        }

        for (Instances split : splitDataNumeric) {
            split.compactify();
        }

        return splitDataNumeric;
    }
}
