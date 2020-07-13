package tsml.contrib.transformers;

import tsml.transformers.Transformer;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import experiments.data.DatasetLoading;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * This class splits a time series into intervals of approximately equal length.
 * Then within each interval, a least squares regression line is calculated
 * and the gradient of this line is returned. Therefore, this transformer
 * will produce a time series of length numIntervals where each element
 * represents the gradient of the line within each interval.
 *
 * @author Vincent Nicholson
 *
 */
public class SlopeTransformer implements Transformer {

    private int numIntervals;

    public SlopeTransformer() {
        this.numIntervals = 8;
    }

    public SlopeTransformer(int numIntervals) {
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
        //remove class attribute if needed
        double[] temp;
        int c = inst.classIndex();
        if(c >= 0) {
            temp=new double[data.length-1];
            System.arraycopy(data,0,temp,0,c); //assumes class attribute is in last index
            data=temp;
        }
        double [] gradients = getGradients(data);
        //Now in DWT form, extract out the terms and set the attributes of new instance
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
            newInstance.setValue(newInstance.numAttributes()-1, inst.classValue());
        return newInstance;
    }

    /**
     * Private function for getting the gradients of a time series.
     *
     * @param inst - the time series to be transformed.
     * @return the transformed inst.
     */
    private double [] getGradients(double [] inst) {
        double [] gradients = new double[this.numIntervals];
        //Split inst into intervals
        double [] [] intervals = getIntervals(inst);
        //perform least squares regression on each interval
        for(int i=0;i<intervals.length;i++) {
            gradients[i] = getGradient(intervals[i]);
        }
        return gradients;
    }

    /**
     * Private function for splitting a time series into
     * approximately equal intervals.
     *
     * @param inst
     * @return
     */
    private double [] [] getIntervals(double [] inst) {
        int numElementsRemaining = inst.length;
        int numIntervalsRemaining = this.numIntervals;
        int startIndex = 0;
        double [] [] intervals = new double [this.numIntervals][];
        for(int i=0;i<this.numIntervals;i++) {
            int intervalSize = (int) Math.ceil(numElementsRemaining/numIntervalsRemaining);
            double [] interval = Arrays.copyOfRange(inst,startIndex,startIndex + intervalSize);
            intervals[i] = interval;
            numElementsRemaining -= intervalSize;
            numIntervalsRemaining--;
            startIndex = startIndex+intervalSize;
        }
        return intervals;
    }

    /**
     * Private method to calculate the gradient of a given interval.
     *
     * @param y - an interval.
     * @return
     */
    private double getGradient(double [] y) {
        double [] x = new double [y.length];
        for(int i=1;i<=y.length;i++) {
            x[i - 1] = i;
        }
        double meanX = calculateMean(x);
        double meanY = calculateMean(y);
        //Calculate w which is given as:
        // w = sum((y-meanY)^2) - sum((x-meanX)^2)
        double ySquaredDiff = 0.0;
        for(int i=0;i<y.length;i++) {
            ySquaredDiff += Math.pow(y[i] - meanY,2);
        }
        double xSquaredDiff = 0.0;
        for(int i=0;i<y.length;i++) {
            xSquaredDiff += Math.pow(x[i] - meanX,2);
        }
        double w = ySquaredDiff - xSquaredDiff;
        // Calculate r which is given as:
        // r = 2*sum((x-meanX)(y-meanY))
        double xyDiff = 0.0;
        for(int i=0;i<y.length;i++) {
            xyDiff += (x[i] - meanX) * (y[i] - meanY);
        }
        double r = 2*xyDiff;
        // The gradient of the least squares regression line.
        // remove NaNs
        double m;
        if (r == 0) {
            m = 0;
        } else {
            m = (w+Math.sqrt(Math.pow(w,2) + Math.pow(r,2))) / r;
        }
        return m;
    }

    /**
     * Private method for calculating the mean of an array.
     *
     * @param arr
     * @return
     */
    private double calculateMean(double [] arr) {
        double sum = 0.0;
        for(int i=0;i<arr.length;i++) {
            sum += arr[i];
        }
        return sum/(double)arr.length;
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
        for(int i = 0; i<numIntervals; i++) {
            attributes.add(new Attribute("SlopeGradient_" + i));
        }
        // Add the class attribute (if it exists)
        if(inputFormat.classIndex() >= 0) {
            attributes.add(inputFormat.classAttribute());
        }
        Instances result = new Instances("Slope" + inputFormat.relationName(), attributes, inputFormat.numInstances());
        // Set the class attribute (if it exists)
        if(inputFormat.classIndex() >= 0) {
            result.setClassIndex(result.numAttributes() - 1);
        }
        return result;
    }
}
