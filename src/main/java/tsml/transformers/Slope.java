package tsml.transformers;

import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * This class splits a time series into intervals of approximately equal length.
 * Then within each interval, a least squares regression line is calculated and
 * the gradient of this line is returned. Therefore, this transformer will
 * produce a time series of length numIntervals where each element represents
 * the gradient of the line within each interval.
 *
 * @author Vincent Nicholson
 *
 */
public class Slope implements Transformer {

    private int numIntervals;

    public Slope() {
        this.numIntervals = 8;
    }

    public Slope(int numIntervals) {
        this.numIntervals = numIntervals;
    }

    public int getNumIntervals() {
        return this.numIntervals;
    }

    public void setNumIntervals(int numIntervals) {
        this.numIntervals = numIntervals;
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
        double[] gradients = getGradients(data);
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


    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        double[][] out = new double[inst.getNumDimensions()][];
        int i =0;
        for(TimeSeries ts : inst){
            checkParameters(ts.getSeriesLength());
            out[i++] = getGradients(ts.toValueArray());
        }

        return new TimeSeriesInstance(out, inst.getLabelIndex(), inst.getClassLabels());
    }

    /**
     * Private function for getting the gradients of a time series.
     *
     * @param inst - the time series to be transformed.
     * @return the transformed inst.
     */
    private double[] getGradients(double[] inst) {
        double[] gradients = new double[this.numIntervals];
        // Split inst into intervals
        double[][] intervals = getIntervals(inst);
        // perform least squares regression on each interval
        for (int i = 0; i < intervals.length; i++) {
            gradients[i] = getGradient(intervals[i]);
        }
        return gradients;
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

    /**
     * Private method to calculate the gradient of a given interval.
     *
     * @param y - an interval.
     * @return
     */
    private double getGradient(double[] y) {
        double[] x = new double[y.length];
        for (int i = 1; i <= y.length; i++) {
            x[i - 1] = i;
        }
        double meanX = calculateMean(x);
        double meanY = calculateMean(y);
        // Calculate w which is given as:
        // w = sum((y-meanY)^2) - sum((x-meanX)^2)
        double ySquaredDiff = 0.0;
        for (int i = 0; i < y.length; i++) {
            ySquaredDiff += Math.pow(y[i] - meanY, 2);
        }
        double xSquaredDiff = 0.0;
        for (int i = 0; i < y.length; i++) {
            xSquaredDiff += Math.pow(x[i] - meanX, 2);
        }
        double w = ySquaredDiff - xSquaredDiff;
        // Calculate r which is given as:
        // r = 2*sum((x-meanX)(y-meanY))
        double xyDiff = 0.0;
        for (int i = 0; i < y.length; i++) {
            xyDiff += (x[i] - meanX) * (y[i] - meanY);
        }
        double r = 2 * xyDiff;
        // The gradient of the least squares regression line.
        // remove NaNs
        double m;
        if (r == 0) {
            m = 0;
        } else {
            m = (w + Math.sqrt(Math.pow(w, 2) + Math.pow(r, 2))) / r;
        }
        return m;
    }

    /**
     * Private method for calculating the mean of an array.
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
        for (int i = 0; i < numIntervals; i++) {
            attributes.add(new Attribute("SlopeGradient_" + i));
        }
        // Add the class attribute (if it exists)
        if (inputFormat.classIndex() >= 0) {
            attributes.add(inputFormat.classAttribute());
        }
        Instances result = new Instances("Slope" + inputFormat.relationName(), attributes, inputFormat.numInstances());
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
    }

    /**
     * Main class for testing.
     * 
     * @param args
     */
    public static void main(String[] args) {
        Instances data = createData(new double[] { 1, 2, 3, 4, 5 });
        // test bad num_intervals (must be at least 1, cannot be higher than the time
        // series length)
        int[] badNumIntervals = new int[] { 0, -5, 6 };
        for (int badNumInterval : badNumIntervals) {
            try {
                Slope s = new Slope(badNumInterval);
                s.transform(data);
                System.out.println("Test failed.");
            } catch (IllegalArgumentException e) {
                System.out.println("Test passed.");
            }
        }
        // test good num_levels
        int[] goodNumIntervals = new int[] { 5, 3, 1 };
        for (int goodNumInterval : goodNumIntervals) {
            try {
                Slope s = new Slope(goodNumInterval);
                s.transform(data);
                System.out.println("Test passed.");
            } catch (IllegalArgumentException e) {
                System.out.println("Test failed.");
            }
        }
        // test output of transformer
        Instances test = createData(new double[] { 4, 6, 10, 12, 8, 6, 5, 5 });
        Slope s = new Slope(2);
        Instances res = s.transform(test);
        double[] resArr = res.get(0).toDoubleArray();
        System.out.println(
                Arrays.equals(resArr, new double[] { (5.0 + Math.sqrt(41)) / 4.0, (1.0 + Math.sqrt(101)) / -10.0 }));

        test = createData(new double[] { -5, 2.5, 1, 3, 10, -1.5, 6, 12, -3, 0.2 });
        res = s.transform(test);
        resArr = res.get(0).toDoubleArray();
        // This is the correct output, but difficult to test if floating point numbers
        // are exactly correct.
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
