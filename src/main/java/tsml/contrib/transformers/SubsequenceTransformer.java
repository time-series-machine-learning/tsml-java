package tsml.contrib.transformers;

import com.sun.org.apache.xpath.internal.operations.Mult;
import tsml.transformers.Transformer;
import utilities.multivariate_tools.MultivariateInstanceTools;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;

public class SubsequenceTransformer implements Transformer {

    private int subsequenceLength;
    private Instances relationalHeader;
    public SubsequenceTransformer(int subsequenceLength) {
        this.subsequenceLength = subsequenceLength;
    }


    @Override
    public Instance transform(Instance inst) {
        double [] timeSeries = inst.toDoubleArray();
        // remove class label
        double[] temp;
        int c = inst.classIndex();
        if(!inst.classIsMissing()) {
            temp=new double[timeSeries.length-1];
            System.arraycopy(timeSeries,0,temp,0,c); //assumes class attribute is in last index
            timeSeries=temp;
        }
        // Extract the subsequences.
        double [] [] subsequences = extractSubsequences(timeSeries);
        //Create the instance - contains only 2 attributes, the relation and the class value.
        Instance newInstance = new DenseInstance(2);
        // Create the relation object.
        Instances relation = createRelation(timeSeries.length);
        // Add the subsequences to the relation object
        for(double [] list : subsequences) {
            Instance newSubsequence = new DenseInstance(1.0,list);
            relation.add(newSubsequence);
        }
        // set the dataset for the newInstance
        newInstance.setDataset(this.relationalHeader);
        // Add the relation to the first attribute.
        int index = newInstance.attribute(0).addRelation(relation);
        newInstance.setValue(0,index);
        // Add the class value
        newInstance.setValue(1,inst.classValue());
        return newInstance;
    }

    /**
     * Method to extract all subsequences given a time series.
     *
     * @param timeSeries - the time series to transform.
     * @return a 2D array of subsequences.
     */
    private double [] [] extractSubsequences(double [] timeSeries) {
        int padAmount = (int) Math.floor(subsequenceLength/2);
        double [] paddedTimeSeries = padTimeSeries(timeSeries,padAmount);
        double [] [] subsequences = new double [timeSeries.length] [subsequenceLength];
        for(int i=0;i<timeSeries.length;i++) {
            subsequences[i] = Arrays.copyOfRange(paddedTimeSeries,i,i+subsequenceLength);
        }
        return subsequences;
    }

    /**
     * Private method for padding the time series on either end by padAmount times.
     * The beginning is padded by the first value of the time series and the end
     * is padded by the last element in the time series.
     *
     * @param timeSeries
     * @param padAmount
     * @return
     */
    private double [] padTimeSeries(double [] timeSeries,int padAmount) {
        double [] newTimeSeries = new double [timeSeries.length + padAmount*2];
        double valueToAdd;
        for(int i=0;i<newTimeSeries.length;i++) {
            // Add the first element
            if(i<padAmount) {
                valueToAdd = timeSeries[0];
            } else if (i>(timeSeries.length)) {
                // Add the last element
                valueToAdd = timeSeries[timeSeries.length-1];
            } else {
                valueToAdd = timeSeries[i-padAmount];
            }
            newTimeSeries[i] = valueToAdd;
        }
        return newTimeSeries;
    }

    @Override
    public Instances determineOutputFormat(Instances inputFormat) throws IllegalArgumentException {
        if (inputFormat.classIndex() != inputFormat.numAttributes() - 1) {
            throw new IllegalArgumentException("cannot handle class values not at end");
        }
        //Set up instances size and format.
        //Create the relation.
        Instances relation = createRelation(inputFormat.numAttributes());
        //Create an arraylist of 2 containing the relational attribute and the class label.
        ArrayList<Attribute> newAtts = new ArrayList<>();
        newAtts.add(new Attribute("relationalAtt", relation));
        newAtts.add(inputFormat.classAttribute());
        //Put them into a new Instances object.
        Instances newFormat = new Instances("Subsequences", newAtts, inputFormat.numInstances());
        newFormat.setRelationName("Subsequences_" + inputFormat.relationName());
        newFormat.setClassIndex(newFormat.numAttributes() - 1);
        this.relationalHeader = newFormat;
        return newFormat;
    }

    /**
     * Private method for creating the relation as it always has the same dimensions -
     * numAtts() = subsequenceLength and numInsts() = timeSeriesLength.
     * @return
     */
    private Instances createRelation(int timeSeriesLength) {
        ArrayList<Attribute> attributes = new ArrayList<>();
        // Add the original elements
        for (int i = 0; i < this.subsequenceLength; i++)
            attributes.add(new Attribute("Subsequence_element_" + i));
        // Number of instances is equal to the length of the time series (as that's the number of
        // subsequences you extract)
        Instances relation = new Instances("Subsequences", attributes, timeSeriesLength);
        return relation;
    }
}
