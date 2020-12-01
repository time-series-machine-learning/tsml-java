package tsml.transformers;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.utilities.TimeSeriesSummaryStatistics;

/**
 * This class is used to extract subsequences (default length = 30) of a time
 * series by extracting neighbouring data points about each data point. For
 * example, if a time series has data points 1,2,3,4,5 and subsequenceLength is
 * 3, This transformer would produce the following output:
 *
 * {{1,1,2},{1,2,3},{2,3,4},{3,4,5},{4,5,5}}
 *
 * Used mainly by ShapeDTW1NN.
 *
 */
public class Subsequences implements Transformer {

    private int subsequenceLength;
    private Instances relationalHeader;
    private boolean normalise = true;
    public Subsequences() {this.subsequenceLength = 30;}

    public Subsequences(int subsequenceLength) {
        this.subsequenceLength = subsequenceLength;
    }

    public void setNormalise(boolean normalise) {this.normalise = normalise;}

    public boolean getNormalise() {return normalise;}

    @Override
    public Instance transform(Instance inst) {
        double[] timeSeries = inst.toDoubleArray();
        checkParameters();
        // remove class label
        double[] temp;
        int c = inst.classIndex();
        if (inst.classIndex() > 0) {
            temp = new double[timeSeries.length - 1];
            System.arraycopy(timeSeries, 0, temp, 0, c); // assumes class attribute is in last index
            timeSeries = temp;
        }
        // Normalise the time series
        double[] temp2;
        if (normalise) {
            temp2 = new double[timeSeries.length];
            System.arraycopy(timeSeries, 0, temp2, 0, timeSeries.length - 1);
            double mean = calculateMean(temp2);
            double sd = calculateSD(temp2, mean);
            timeSeries = normaliseArray(temp2, mean, sd);
        }
        // Extract the subsequences.
        double[][] subsequences = extractSubsequences(timeSeries);
        // Create the instance - contains only 2 attributes, the relation and the class
        // value.
        Instance newInstance = new DenseInstance(2);
        // Create the relation object.
        Instances relation = createRelation(timeSeries.length);
        // Add the subsequences to the relation object
        for (double[] list : subsequences) {
            Instance newSubsequence = new DenseInstance(1.0, list);
            relation.add(newSubsequence);
        }
        // set the dataset for the newInstance
        newInstance.setDataset(this.relationalHeader);
        // Add the relation to the first attribute.
        int index = newInstance.attribute(0).addRelation(relation);
        newInstance.setValue(0, index);
        // Add the class value
        newInstance.setValue(1, inst.classValue());
        return newInstance;
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        double[][] out = new double[inst.getNumDimensions()][];
        int i =0;
        for(TimeSeries ts : inst){
            // Extract the subsequences.
            double[][] subsequences = extractSubsequences(TimeSeriesSummaryStatistics.standardNorm(ts).toValueArray());
            
            //stack subsequences if we're multivariate.
            for(double[] d : subsequences)
                out[i++] = d;
        }

        return new TimeSeriesInstance(out, inst.getLabelIndex(), inst.getClassLabels());
    }

    /**
     * Method to extract all subsequences given a time series.
     *
     * @param timeSeries - the time series to transform.
     * @return a 2D array of subsequences.
     */
    private double[][] extractSubsequences(double[] timeSeries) {
        int padAmount = (int) Math.floor(subsequenceLength / 2);
        double[] paddedTimeSeries = padTimeSeries(timeSeries, padAmount);
        double[][] subsequences = new double[timeSeries.length][subsequenceLength];
        for (int i = 0; i < timeSeries.length; i++) {
            subsequences[i] = Arrays.copyOfRange(paddedTimeSeries, i, i + subsequenceLength);
        }
        return subsequences;
    }

    /**
     * Private method for padding the time series on either end by padAmount times.
     * The beginning is padded by the first value of the time series and the end is
     * padded by the last element in the time series.
     *
     * @param timeSeries
     * @param padAmount
     * @return
     */
    private double[] padTimeSeries(double[] timeSeries, int padAmount) {
        double[] newTimeSeries = new double[timeSeries.length + padAmount * 2];
        double valueToAdd;
        for (int i = 0; i < newTimeSeries.length; i++) {
            // Add the first element
            if (i < padAmount) {
                valueToAdd = timeSeries[0];
            } else if (i >= padAmount && i < (padAmount + timeSeries.length)) {
                // Add the original time series element
                valueToAdd = timeSeries[i - padAmount];
            } else {
                // Add the last element
                valueToAdd = timeSeries[timeSeries.length - 1];
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
        // Set up instances size and format.
        // Create the relation.
        Instances relation = createRelation(inputFormat.numAttributes() - 1);
        // Create an arraylist of 2 containing the relational attribute and the class
        // label.
        ArrayList<Attribute> newAtts = new ArrayList<>();
        newAtts.add(new Attribute("relationalAtt", relation));
        newAtts.add(inputFormat.classAttribute());
        // Put them into a new Instances object.
        Instances newFormat = new Instances("Subsequences", newAtts, inputFormat.numInstances());
        newFormat.setRelationName("Subsequences_" + inputFormat.relationName());
        newFormat.setClassIndex(newFormat.numAttributes() - 1);
        this.relationalHeader = newFormat;
        return newFormat;
    }

    /**
     * Private function to calculate the mean of an array.
     *
     * @param arr
     * @return
     */
    private double calculateMean(double[] arr) {
        double sum = 0.0;
        for (int i = 0; i < arr.length; i++) {
            sum += arr[i];
        }
        return sum / (double) arr.length;
    }

    /**
     * Private function to calculate the standard deviation of an array.
     *
     * @param arr
     * @return
     */
    private double calculateSD(double[] arr, double mean) {
        double sum = 0.0;
        for (int i = 0; i < arr.length; i++) {
            sum += Math.pow(arr[i] - mean, 2);
        }
        sum = sum / (double) arr.length;
        return Math.sqrt(sum);
    }

    /**
     * Private function for normalising an array.
     *
     * @param arr
     * @param mean
     * @param sd
     * @return
     */
    private double[] normaliseArray(double[] arr, double mean, double sd) {
        double[] normalisedArray = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            normalisedArray[i] = (arr[i] - mean) / sd;
        }
        return normalisedArray;
    }

    /**
     * Private method for creating the relation as it always has the same dimensions
     * - numAtts() = subsequenceLength and numInsts() = timeSeriesLength.
     * 
     * @return
     */
    private Instances createRelation(int timeSeriesLength) {
        ArrayList<Attribute> attributes = new ArrayList<>();
        // Add the original elements
        for (int i = 0; i < this.subsequenceLength; i++)
            attributes.add(new Attribute("Subsequence_element_" + i));
        // Number of instances is equal to the length of the time series (as that's the
        // number of
        // subsequences you extract)
        Instances relation = new Instances("Subsequences", attributes, timeSeriesLength);
        return relation;
    }

    private void checkParameters() {
        if (this.subsequenceLength < 1) {
            throw new IllegalArgumentException("subsequenceLength cannot be less than 1.");
        }
    }

    public static void main(String[] args) {
        Instances data = createData();

        // test bad SubsequenceLength
        // has to be greater than 0.
        int[] badSubsequenceLengths = new int[] { -1, 0, -99999999 };
        for (int badSubsequenceLength : badSubsequenceLengths) {
            try {
                Subsequences s = new Subsequences(badSubsequenceLength);
                s.transform(data);
                System.out.println("Test failed.");
            } catch (IllegalArgumentException e) {
                System.out.println("Test passed.");
            }
        }
        // test good SubsequenceLength
        int[] goodSubsequenceLengths = new int[] { 1, 50, 999 };
        for (int goodSubsequenceLength : goodSubsequenceLengths) {
            try {
                Subsequences s = new Subsequences(goodSubsequenceLength);
                s.transform(data);
                System.out.println("Test passed.");
            } catch (IllegalArgumentException e) {
                System.out.println("Test failed.");
            }
        }
        // check output
        Subsequences s = new Subsequences(5);
        Instances res = s.transform(data);
        Instances inst = res.get(0).relationalValue(0);
        double[] resArr = inst.get(0).toDoubleArray();
        System.out.println(Arrays.equals(resArr, new double[] { 1, 1, 1, 2, 3 }));
        resArr = inst.get(1).toDoubleArray();
        System.out.println(Arrays.equals(resArr, new double[] { 1, 1, 2, 3, 4 }));
        resArr = inst.get(2).toDoubleArray();
        System.out.println(Arrays.equals(resArr, new double[] { 1, 2, 3, 4, 5 }));
        resArr = inst.get(3).toDoubleArray();
        System.out.println(Arrays.equals(resArr, new double[] { 2, 3, 4, 5, 5 }));
        resArr = inst.get(4).toDoubleArray();
        System.out.println(Arrays.equals(resArr, new double[] { 3, 4, 5, 5, 5 }));
    }

    /**
     * Function to create data for testing purposes.
     *
     * @return
     */
    private static Instances createData() {
        // Create the attributes
        ArrayList<Attribute> atts = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            atts.add(new Attribute("test_" + i));
        }
        // Create the class values
        ArrayList<String> classes = new ArrayList<>();
        classes.add("1");
        classes.add("0");
        atts.add(new Attribute("class", classes));
        Instances newInsts = new Instances("Test_dataset", atts, 5);
        newInsts.setClassIndex(newInsts.numAttributes() - 1);

        // create the test data
        double[] test = new double[] { 1, 2, 3, 4, 5 };
        createInst(test, "1", newInsts);
        test = new double[] { 1, 1, 2, 3, 4 };
        createInst(test, "1", newInsts);
        test = new double[] { 2, 2, 2, 3, 4 };
        createInst(test, "0", newInsts);
        test = new double[] { 2, 3, 4, 5, 6 };
        createInst(test, "0", newInsts);
        test = new double[] { 0, 1, 1, 1, 2 };
        createInst(test, "1", newInsts);
        return newInsts;
    }

    /**
     * private function for creating an instance from a double array. Used for
     * testing purposes.
     *
     * @param arr
     * @return
     */
    private static void createInst(double[] arr, String classValue, Instances dataset) {
        Instance inst = new DenseInstance(arr.length + 1);
        for (int i = 0; i < arr.length; i++) {
            inst.setValue(i, arr[i]);
        }
        inst.setDataset(dataset);
        inst.setClassValue(classValue);
        dataset.add(inst);
    }


}
