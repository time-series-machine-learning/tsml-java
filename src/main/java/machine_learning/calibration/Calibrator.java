package machine_learning.calibration;

import evaluation.storage.ClassifierResults;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;

public interface Calibrator {

    public void buildCalibrator(TimeSeriesInstances classifierProbs) throws Exception;

    default public void buildCalibrator(ClassifierResults classifierResults) throws Exception {
        double[][] classifierProbs = classifierResults.getProbabilityDistributionsAsArray();
        double[] classVals = classifierResults.getTrueClassValsAsArray();

        double[][][] tsdata = new double[classifierProbs.length][][];
        int[] labelIndices = new int[classifierProbs.length];

        for (int i = 0; i < classifierProbs.length; i++) {
            tsdata[i] = new double[][] { classifierProbs[i] };
            labelIndices[i] = (int)classVals[i];
        }

        TimeSeriesInstances ts = new TimeSeriesInstances(tsdata, labelIndices);

        //build dummy string class labels, needed later in transforms. indices not enough
        String[] t = new String[classifierResults.numClasses()];
        for (int i = 0; i < classifierResults.numClasses(); i++) {
            t[i] = String.valueOf(i);
        }
        ts.setClassLabels(t);

        buildCalibrator(ts);
    }





    public double[] calibrateInstance(TimeSeriesInstance classifierProbs) throws Exception;

    default public double[][] calibrateInstances(TimeSeriesInstances classifierProbs) throws Exception {
        double[][] calibratedProbs = new double[classifierProbs.numInstances()][];
        for (int i = 0; i < classifierProbs.numInstances(); i++)
            calibratedProbs[i] = calibrateInstance(classifierProbs.get(i));

        return calibratedProbs;
    }

    default public double[] calibrateInstance(double[] classifierProbs) throws Exception {
        return calibrateInstance(new TimeSeriesInstance(new double[][] { classifierProbs }));
    }

    default public double[][] calibrateInstances(ClassifierResults classifierProbs) throws Exception {
        double[][] calibratedProbs = new double[classifierProbs.numInstances()][];
        for (int i = 0; i < classifierProbs.numInstances(); i++)
            calibratedProbs[i] = calibrateInstance(classifierProbs.getProbabilityDistribution(i));

        return calibratedProbs;
    }

}
