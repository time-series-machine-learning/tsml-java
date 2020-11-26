package tsml.transformers;

import org.apache.commons.lang3.ArrayUtils;

import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * This class performs a Haar Wavelet transformation on a given time series. The
 * result is the approximate coefficients of the highest level and the wavelet
 * coefficients of every level from numLevels down to one are all concatenated
 * together.
 *
 * @author Vincent Nicholson
 *
 */
public class DWT implements Transformer {

    private int numLevels;

    public DWT() {
        this.numLevels = 3;
    }

    public DWT(int numLevels) {
        this.numLevels = numLevels;
    }

    public int getNumLevels() {
        return this.numLevels;
    }

    public void setNumLevels(int numLevels) {
        this.numLevels = numLevels;
    }

    @Override
    public Instance transform(Instance inst) {
        checkParameters();
        double[] data = inst.toDoubleArray();
        // remove class attribute if needed
        double[] temp;
        int c = inst.classIndex();
        if (c >= 0) {
            temp = new double[data.length - 1];
            System.arraycopy(data, 0, temp, 0, c); // assumes class attribute is in last index
            data = temp;
        }
        double[] waveletCoeffs = getDWTCoefficients(data);
        // Now in DWT form, extract out the terms and set the attributes of new instance
        Instance newInstance;
        int numAtts = waveletCoeffs.length;
        if (inst.classIndex() >= 0)
            newInstance = new DenseInstance(numAtts + 1);
        else
            newInstance = new DenseInstance(numAtts);
        // Copy over the values into the Instance
        for (int j = 0; j < numAtts; j++)
            newInstance.setValue(j, waveletCoeffs[j]);
        // Set the class value
        if (inst.classIndex() >= 0)
            newInstance.setValue(newInstance.numAttributes() - 1, inst.classValue());

        return newInstance;
    }

    /**
     * Private function for calculating the wavelet coefficients of a given time
     * series.
     *
     * @param inst - the time series to be transformed.
     * @return the transformed inst.
     */
    private double[] getDWTCoefficients(double[] inst) {
        // For temporary storage of each array
        double[][] vectors = new double[this.numLevels + 1][];
        if (numLevels == 0) {
            return inst;
        } else {
            // Extract the coefficients on each level
            double[] current = inst;
            for (int i = 0; i < numLevels; i++) {
                double[] approxCoeffs = getApproxCoefficients(current);
                double[] waveletCoeffs = getWaveletCoefficients(current);
                vectors[i] = waveletCoeffs;
                current = approxCoeffs;
            }
            vectors[numLevels] = current;
        }
        // Combine the double array into one.
        return concatenateVectors(vectors);
    }

    /**
     * Private method for combining the 2d array of vectors into the correct order.
     *
     * @param vectors
     * @return
     */
    private double[] concatenateVectors(double[][] vectors) {
        double[] out = new double[] {};
        for (int i = vectors.length - 1; i > -1; i--) {
            out = ArrayUtils.addAll(out, vectors[i]);
        }
        return out;
    }

    /**
     * Private method to calculate the approximate coefficients of a time series t.
     *
     * @param t - the time series.
     * @return
     */
    public double[] getApproxCoefficients(double[] t) {
        if (t.length == 1) {
            return t;
        }
        int total = (int) Math.floor(t.length / 2);
        double[] coeffs = new double[total];
        for (int i = 0; i < total; i++) {
            coeffs[i] = ((t[2 * i] + t[2 * i + 1]) / Math.sqrt(2));
        }
        return coeffs;
    }

    /**
     * Private method to calculate the wavelet coefficients of a time series t.
     *
     * @param t - the time series.
     * @return
     */
    public double[] getWaveletCoefficients(double[] t) {
        if (t.length == 1) {
            return t;
        }
        int total = (int) Math.floor(t.length / 2);
        double[] coeffs = new double[total];
        for (int i = 0; i < total; i++) {
            coeffs[i] = ((t[2 * i] - t[2 * i + 1]) / Math.sqrt(2));
        }
        return coeffs;
    }
    
    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        //could do this across all dimensions.
        double[][] out = new double[inst.getNumDimensions()][];
        int i = 0;
        for(TimeSeries ts : inst){
            out[i++] = getDWTCoefficients(ts.toValueArray());
        }
        
        //create a new output instance with the ACF data.
        return new TimeSeriesInstance(out, inst.getLabelIndex(), inst.getClassLabels());
    }

    @Override
    public Instances determineOutputFormat(Instances inputFormat) throws IllegalArgumentException {
        // If the class index exists.
        if (inputFormat.classIndex() >= 0) {
            if (inputFormat.classIndex() != inputFormat.numAttributes() - 1) {
                throw new IllegalArgumentException("cannot handle class values not at end");
            }
        }
        int numAttributes = calculateNumAttributes(inputFormat.numAttributes());
        ArrayList<Attribute> attributes = new ArrayList<>();
        // Create a list of attributes
        for (int i = 0; i < numAttributes; i++) {
            attributes.add(new Attribute("DWTCoefficient_" + i));
        }
        // Add the class attribute (if it exists)
        if (inputFormat.classIndex() >= 0) {
            attributes.add(inputFormat.classAttribute());
        }
        Instances result = new Instances("DWT" + inputFormat.relationName(), attributes, inputFormat.numInstances());
        // Set the class attribute (if it exists)
        if (inputFormat.classIndex() >= 0) {
            result.setClassIndex(result.numAttributes() - 1);
        }
        return result;
    }

    /**
     * Private method to calculate the number of attributes produced by DWT.
     *
     * @return - int, the number of attributes that DWT produces.
     */
    private int calculateNumAttributes(int timeSeriesLength) {
        int numLevels = this.numLevels;
        if (numLevels == 0) {
            return timeSeriesLength;
        } else {
            int counter = 0;
            // Record the length of the time series at the current level
            int timeSeriesLengthAtCurLevel = timeSeriesLength;
            for (int i = 0; i < numLevels; i++) {
                if (timeSeriesLengthAtCurLevel != 1) {
                    timeSeriesLengthAtCurLevel = (int) Math.floor(timeSeriesLengthAtCurLevel / 2);
                }
                counter += timeSeriesLengthAtCurLevel;
                // If at the last level
                if (i == numLevels - 1) {
                    counter += timeSeriesLengthAtCurLevel;
                }
            }
            return counter;
        }
    }

    private void checkParameters() {
        if (this.numLevels < 0) {
            throw new IllegalArgumentException("numLevels cannot be negative.");
        }
    }

    public static void main(String[] args) throws Exception {
        Instances data = createData(new double[] { 1, 2, 3, 4, 5 });
        // test bad num_levels
        // num_levels cannot be negative
        int[] badNumLevels = new int[] { -1, -5, -999 };
        for (int badNumLevel : badNumLevels) {
            try {
                DWT d = new DWT(badNumLevel);
                d.transform(data);
                System.out.println("Test failed.");
            } catch (IllegalArgumentException e) {
                System.out.println("Test passed.");
            }
        }
        // test good num_levels
        int[] goodNumLevels = new int[] { 0, 1, 999 };
        for (int goodNumLevel : goodNumLevels) {
            try {
                DWT d = new DWT(goodNumLevel);
                d.transform(data);
                System.out.println("Test passed.");
            } catch (IllegalArgumentException e) {
                System.out.println("Test failed.");
            }
        }

        // check output
        data = createData(new double[] { 4, 6, 10, 12, 8, 6, 5, 5 });
        DWT d = new DWT(2);
        double[] resArr = d.transform(data).get(0).toDoubleArray();
        // This is equivalent but rounding errors prevent from checking exactly
        System.out.println(
                Arrays.equals(resArr, new double[] { 16, 12, -6, 2, -Math.sqrt(2), -Math.sqrt(2), Math.sqrt(2), 0 }));
        data = createData(new double[] { -5, 2.5, 1, 3, 10, -1.5, 6, 12, -3 });
        resArr = d.transform(data).get(0).toDoubleArray();
        // Same issue here
        System.out.println(
                Arrays.equals(resArr, new double[] { 0.75, 13.25, -3.25, -4.75, -5.303, -1.414, 8.132, -4.243 }));
        // check that num levels being zero does no change
        data = createData(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        d = new DWT(0);
        resArr = d.transform(data).get(0).toDoubleArray();
        System.out.println(Arrays.equals(data.get(0).toDoubleArray(), resArr));
    }

    /**
     * Function to create data for testing purposes.
     *
     * @return
     */
    private static Instances createData(double[] data) {
        // Create the attributes
        ArrayList<Attribute> atts = new ArrayList<>();
        for (int i = 0; i < data.length; i++) {
            atts.add(new Attribute("test_" + i));
        }
        Instances newInsts = new Instances("Test_dataset", atts, 1);
        // create the test data
        createInst(data, newInsts);
        return newInsts;
    }

    /**
     * private function for creating an instance from a double array. Used for
     * testing purposes.
     *
     * @param arr
     * @return
     */
    private static void createInst(double[] arr, Instances dataset) {
        Instance inst = new DenseInstance(arr.length);
        for (int i = 0; i < arr.length; i++) {
            inst.setValue(i, arr[i]);
        }
        inst.setDataset(dataset);
        dataset.add(inst);
    }


}
