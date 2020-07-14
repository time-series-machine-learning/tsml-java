package tsml.transformers;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * This class is used to extract subsequences (default length = 30) of a time series by extracting
 * neighbouring data points about each data point. For example, if a time
 * series has data points 1,2,3,4,5 and subsequenceLength is 3, This transformer
 * would produce the following output:
 *
 * {{1,1,2},{1,2,3},{2,3,4},{3,4,5},{4,5,5}}
 *
 * Used mainly by ShapeDTW1NN.
 *
 */
public class Subsequences implements Transformer {

    private int subsequenceLength;
    private Instances relationalHeader;
    public Subsequences() {this.subsequenceLength = 30;}
    public Subsequences(int subsequenceLength) {
        this.subsequenceLength = subsequenceLength;
    }

    @Override
    public Instance transform(Instance inst) {
        double [] timeSeries = inst.toDoubleArray();
        checkParameters(inst)
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
        Instances relation = createRelation(inputFormat.numAttributes()-1);
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

    public static void main(String[] args) {
        Instances data = createData();

        //test bad SubsequenceLength
        int [] badSubsequenceLength
        //test good SubsequenceLength

        //check output

        //check dimensions
    }

    /**
     * Function to create data for testing purposes.
     *
     * @return
     */
    private static Instances createData() {
        //Create the attributes
        ArrayList<Attribute> atts = new ArrayList<>();
        for(int i=0;i<5;i++) {
            atts.add(new Attribute("test_" + i));
        }
        //Create the class values
        ArrayList<String> classes = new ArrayList<>();
        classes.add("1");
        classes.add("0");
        atts.add(new Attribute("class",classes));
        Instances newInsts = new Instances("Test_dataset",atts,5);
        newInsts.setClassIndex(newInsts.numAttributes()-1);

        //create the test data
        double [] test = new double [] {1,2,3,4,5};
        createInst(test,"1",newInsts);
        test = new double [] {1,1,2,3,4};
        createInst(test,"1",newInsts);
        test = new double [] {2,2,2,3,4};
        createInst(test,"0",newInsts);
        test = new double [] {2,3,4,5,6};
        createInst(test,"0",newInsts);
        test = new double [] {0,1,1,1,2};
        createInst(test,"1",newInsts);
        return newInsts;
    }

    /**
     * private function for creating an instance from a double array. Used
     * for testing purposes.
     *
     * @param arr
     * @return
     */
    private static void createInst(double [] arr,String classValue, Instances dataset) {
        Instance inst = new DenseInstance(arr.length+1);
        for(int i=0;i<arr.length;i++) {
            inst.setValue(i,arr[i]);
        }
        inst.setDataset(dataset);
        inst.setClassValue(classValue);
        dataset.add(inst);
    }
}
