package tsml.transformers;

import tsml.classifiers.distance_based.ShapeDTW_1NN;
import tsml.data_containers.TimeSeriesInstance;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

/*
 * This class is used to transform a time series into a set
 * of ShapeDTW distances. It does this by calculating the
 * distance between an unknown time series and all
 * time series in a reference set.
 */
public class ShapeDTWFeatures implements Transformer {

    private Instances referenceSet;
    private Instances dataset;

    /**
     * No default parameter because the reference set must be given inorder
     * to be able to calculate distances.
     *
     * @param referenceSet
     */
    public ShapeDTWFeatures(Instances referenceSet) {
        this.referenceSet = referenceSet;
    }

    @Override
    public Instance transform(Instance inst) {
        double [] distances = getDistances(inst);
        //Now in ShapeDTW distances form, extract out the terms and set the attributes of new instance
        Instance newInstance;
        int numAtts = distances.length;
        if (inst.classIndex() >= 0)
            newInstance = new DenseInstance(numAtts + 1);
        else
            newInstance = new DenseInstance(numAtts);
        // Copy over the values into the Instance
        for (int j = 0; j < numAtts; j++)
            newInstance.setValue(j, distances[j]);
        // Set the class value
        if (inst.classIndex() >= 0)
            newInstance.setValue(newInstance.numAttributes()-1, inst.classValue());
        newInstance.setDataset(dataset);
        return newInstance;
    }

    /**
     * Private function to get all the distances from inst and all the
     * time series in the reference set.
     *
     * @param inst
     * @return
     */
    private double [] getDistances(Instance inst) {
        double [] dists = new double[this.referenceSet.numInstances()];
        for(int i=0;i< dists.length;i++) {
            dists[i] = ShapeDTW_1NN.NN_DTW_Subsequences.calculateDistance(inst,referenceSet.get(i));
        }
        return dists;
    }

    @Override
    public Instances determineOutputFormat(Instances inputFormat) throws IllegalArgumentException {
        //If the class index exists.
        if(inputFormat.classIndex() >= 0) {
            if (inputFormat.classIndex() != inputFormat.numAttributes() - 1) {
                throw new IllegalArgumentException("cannot handle class values not at end");
            }
        }
        ArrayList<Attribute> attributes = new ArrayList<>();
        // Create a list of attributes
        for(int i = 0; i<referenceSet.numInstances(); i++) {
            attributes.add(new Attribute("ShapeDTW_Distance_" + i));
        }
        // Add the class attribute (if it exists)
        if(inputFormat.classIndex() >= 0) {
            attributes.add(inputFormat.classAttribute());
        }
        Instances result = new Instances("ShapeDTWDistances" + inputFormat.relationName(), attributes, inputFormat.numInstances());
        // Set the class attribute (if it exists)
        if(inputFormat.classIndex() >= 0) {
            result.setClassIndex(result.numAttributes() - 1);
        }
        this.dataset = result;
        return result;
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        return null;
    }
}
