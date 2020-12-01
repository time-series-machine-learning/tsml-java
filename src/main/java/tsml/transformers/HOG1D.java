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
 * This class is to calculate the HOG1D transform of a dataframe of time series
 * data. Works by splitting the time series num_intervals times, and calculate a
 * histogram of gradients within each interval.
 *
 * @author Vincent Nicholson
 *
 */
public class HOG1D implements Transformer {

    private int numIntervals;
    private int numBins;
    private double scalingFactor;

    public HOG1D() {
        this.numIntervals = 2;
        this.numBins = 8;
        this.scalingFactor = 0.1;
    }

    public HOG1D(int numIntervals, int numBins, double scalingFactor) {
        this.numIntervals = numIntervals;
        this.numBins = numBins;
        this.scalingFactor = scalingFactor;
    }

    public int getNumIntervals() {
        return this.numIntervals;
    }

    public void setNumIntervals(int numIntervals) {
        this.numIntervals = numIntervals;
    }

    public int getNumBins() {
        return this.numBins;
    }

    public void setNumBins(int numBins) {
        this.numBins = numBins;
    }

    public double getScalingFactor() {
        return scalingFactor;
    }

    public void setScalingFactor(double scalingFactor) {
        this.scalingFactor = scalingFactor;
    }

    @Override
    public Instance transform(Instance inst) {
        double[] data = inst.toDoubleArray();
        // remove class attribute if needed
        double[] temp;
        int c = inst.classIndex();
        if (c >= 0) {
            temp = new double[data.length - 1];
            System.arraycopy(data, 0, temp, 0, c); // assumes class attribute is in last index
            data = temp;
        }
        checkParameters(data.length);
        double[] gradients = getHOG1Ds(data);
        // Now in DWT form, extract out the terms and set the attributes of new instance
        Instance newInstance;
        int numAtts = gradients.length;
        if (inst.classIndex() >= 0)
            newInstance = new DenseInstance(numAtts + 1);
        else
            newInstance = new DenseInstance(numAtts);
        // Copy over the values into the Instance
        for (int j = 0; j < numAtts; j++)
            newInstance.setValue(j, gradients[j]);
        // Set the class value
        if (inst.classIndex() >= 0)
            newInstance.setValue(newInstance.numAttributes() - 1, inst.classValue());
        return newInstance;
    }

    /**
     * Private function for getting the histogram of gradients of a time series.
     *
     * @param inst - the time series to be transformed.
     * @return the transformed inst.
     */
    private double[] getHOG1Ds(double[] inst) {
        double[][] hog1Ds = new double[this.numIntervals][];
        // Split inst into intervals
        double[][] intervals = getIntervals(inst);
        // Extract a histogram of gradients for each interval
        for (int i = 0; i < intervals.length; i++) {
            hog1Ds[i] = getHOG1D(intervals[i]);
        }
        // Concatenate the HOG1Ds together
        double[] out = new double[] {};
        for (int i = hog1Ds.length - 1; i > -1; i--) {
            out = ArrayUtils.addAll(hog1Ds[i], out);
        }
        return out;
    }

    /**
     * Private function for splitting a time series into approximately equal
     * intervals.
     *
     * @param inst
     * @return
     */
    private double[][] getIntervals(double[] inst) {
        int numElementsRemaining = inst.length;
        int numIntervalsRemaining = this.numIntervals;
        int startIndex = 0;
        double[][] intervals = new double[this.numIntervals][];
        for (int i = 0; i < this.numIntervals; i++) {
            int intervalSize = (int) Math.ceil(numElementsRemaining / numIntervalsRemaining);
            double[] interval = Arrays.copyOfRange(inst, startIndex, startIndex + intervalSize);
            intervals[i] = interval;
            numElementsRemaining -= intervalSize;
            numIntervalsRemaining--;
            startIndex = startIndex + intervalSize;
        }
        return intervals;
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        double[][] out = new double[inst.getNumDimensions()][];
        int i =0;
        for(TimeSeries ts : inst){
            out[i++] = getHOG1Ds(ts.toValueArray());
        }

        return new TimeSeriesInstance(out, inst.getLabelIndex(), inst.getClassLabels());
    }

    /**
     * Private method to calculate the HOG of a particular interval.
     *
     * @param t - an interval.
     * @return
     */
    private double[] getHOG1D(double[] t) {
        // Pad t on either ends just once.
        double[] paddedT = ArrayUtils.addAll(new double[] { t[0] }, t);
        paddedT = ArrayUtils.addAll(paddedT, new double[] { t[t.length - 1] });
        // Calculate the gradients over every element in t.
        double[] gradients = new double[t.length];
        for (int i = 1; i < gradients.length + 1; i++) {
            gradients[(i - 1)] = scalingFactor * 0.5 * (paddedT[(i + 1)] - paddedT[(i - 1)]);
        }
        // Then, calculate the orientations given the gradients
        double[] orientations = new double[gradients.length];
        for (int i = 0; i < gradients.length; i++) {
            orientations[i] = Math.toDegrees(Math.atan(gradients[i]));
        }
        double[] histBins = new double[this.numBins];
        // Calculate the bin boundaries
        double inc = 180.0 / (double) this.numBins;
        double current = -90.0;
        for (int i = 0; i < histBins.length; i++) {
            histBins[i] = current + inc;
            current += inc;
        }
        // Create the histogram
        double[] histogram = new double[this.numBins];
        for (int i = 0; i < orientations.length; i++) {
            for (int j = 0; j < histogram.length; j++) {
                if (orientations[i] <= histBins[j]) {
                    histogram[j] += 1;
                    break;
                }
            }
        }
        return histogram;
    }

    @Override
    public Instances determineOutputFormat(Instances inputFormat) throws IllegalArgumentException {
        // If the class index exists.
        if (inputFormat.classIndex() >= 0) {
            if (inputFormat.classIndex() != inputFormat.numAttributes() - 1) {
                throw new IllegalArgumentException("cannot handle class values not at end");
            }
        }
        ArrayList<Attribute> attributes = new ArrayList<>();
        // Create a list of attributes
        for (int i = 0; i < numBins * numIntervals; i++) {
            attributes.add(new Attribute("HOG1D_" + i));
        }
        // Add the class attribute (if it exists)
        if (inputFormat.classIndex() >= 0) {
            attributes.add(inputFormat.classAttribute());
        }
        Instances result = new Instances("HOG1D" + inputFormat.relationName(), attributes, inputFormat.numInstances());
        // Set the class attribute (if it exists)
        if (inputFormat.classIndex() >= 0) {
            result.setClassIndex(result.numAttributes() - 1);
        }
        return result;
    }

    private void checkParameters(int timeSeriesLength) {
        if (this.numIntervals < 1) {
            throw new IllegalArgumentException("numIntervals must be greater than zero.");
        }
        if (this.numIntervals > timeSeriesLength) {
            throw new IllegalArgumentException("numIntervals cannot be longer than the time series length.");
        }
        if (this.numBins < 1) {
            throw new IllegalArgumentException("numBins must be greater than zero.");
        }
    }

    /**
     * Main class for testing.
     *
     * @param args
     */
    public static void main(String[] args) {
        Instances data = createData(new double[] { 4, 6, 10, 12, 8, 6, 5, 5 });

        // test bad numIntervals
        // has to be greater than 0.
        // cannot be higher than the time series length
        int[] badNumIntervals = new int[] { -1, 0, -99999999, 9 };
        for (int badNumInterval : badNumIntervals) {
            try {
                HOG1D h = new HOG1D(badNumInterval, 8, 0.1);
                h.transform(data);
                System.out.println("Test failed.");
            } catch (IllegalArgumentException e) {
                System.out.println("Test passed.");
            }
        }
        // test good numIntervals
        int[] goodNumIntervals = new int[] { 2, 4, 8 };
        for (int goodNumInterval : goodNumIntervals) {
            try {
                HOG1D h = new HOG1D(goodNumInterval, 8, 0.1);
                h.transform(data);
                System.out.println("Test passed.");
            } catch (IllegalArgumentException e) {
                System.out.println("Test failed.");
            }
        }
        // test bad numBins
        // Cannot be less than 1
        int[] badNumBins = new int[] { 0, -5, -999, -687 };
        for (int badNumBin : badNumBins) {
            try {
                HOG1D h = new HOG1D(2, badNumBin, 0.1);
                h.transform(data);
                System.out.println("Test failed.");
            } catch (IllegalArgumentException e) {
                System.out.println("Test passed.");
            }
        }
        // test good numBins
        int[] goodNumBins = new int[] { 1, 5, 12, 200 };
        for (int goodNumBin : goodNumBins) {
            try {
                HOG1D h = new HOG1D(2, goodNumBin, 0.1);
                h.transform(data);
                System.out.println("Test passed.");
            } catch (IllegalArgumentException e) {
                System.out.println("Test failed.");
            }
        }
        // test output
        HOG1D h = new HOG1D();
        Instances out = h.transform(data);
        double[] outArr = out.get(0).toDoubleArray();
        System.out.println(Arrays.equals(outArr, new double[] { 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0 }));

        data = createData(new double[] { -5, 2.5, 1, 3, 10, -1.5, 6, 12, -3, 0.2 });
        out = h.transform(data);
        outArr = out.get(0).toDoubleArray();
        System.out.println(Arrays.equals(outArr,
                new double[] { 0.0, 0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0 }));
        // test dimensions (test that output is always of length numBins*numIntervals)
        h = new HOG1D(6, 30, 0.1);
        out = h.transform(data);
        System.out.println(out.get(0).toDoubleArray().length == 6 * 30);
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
