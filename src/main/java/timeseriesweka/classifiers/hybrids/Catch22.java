package timeseriesweka.classifiers.hybrids;

import experiments.data.DatasetLoading;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

import static utilities.ArrayUtilities.mean;
import static utilities.ClusteringUtilities.zNormalise;
import static utilities.GenericTools.max;
import static utilities.GenericTools.min;
import static utilities.InstanceTools.resampleTrainAndTestInstances;
import static utilities.StatisticalUtilities.median;
import static utilities.Utilities.argMax;
import static utilities.Utilities.extractTimeSeries;

public class Catch22 extends AbstractClassifier {

    private Classifier cls = new J48();
    private boolean norm = true;

    private Instances header;

    public Catch22(){}

    @Override
    public void buildClassifier(Instances data) throws Exception {
        ArrayList<Attribute> atts = new ArrayList();
        for (int i = 0; i < 22; i++){
            atts.add(new Attribute("att" + i));
        }
        atts.add(data.classAttribute());
        Instances transformedData = new Instances("Catch22Transform", atts, data.numInstances());
        transformedData.setClassIndex(transformedData.numAttributes()-1);
        header = new Instances(transformedData);

        if (norm){
            data = new Instances(data);
            zNormalise(data);
        }

        for (Instance inst : data){
            transformedData.add(singleTransform(inst, inst.classValue()));
        }

        cls.buildClassifier(transformedData);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        Instance transformedInst = singleTransform(instance, -1);
        transformedInst.setDataset(header);
        return cls.classifyInstance(transformedInst);
    }

    public double[] distributionForInstance(Instance instance) throws Exception {
        Instance transformedInst = singleTransform(instance, -1);
        transformedInst.setDataset(header);
        return cls.distributionForInstance(transformedInst);
    }

    public static Instance singleTransform(Instance inst, double classVal){
        double[] featureSet = new double[23];
        double[] arr = extractTimeSeries(inst);

        featureSet[0] = histMode5DN(arr);
        featureSet[1] = histMode10DN(arr);
        featureSet[2] = binaryStatsMeanLongstretch1SB(arr);
        featureSet[3] = outlierIncludeP001mdrmdDN(arr);
        featureSet[4] = outlierIncludeN001mdrmdDN(arr);

        featureSet[22] = classVal;

        return new DenseInstance(1, featureSet);
    }

    private static double histMode5DN(double[] arr){
        return histogramMode(arr, 5);
    }

    private static double histMode10DN(double[] arr){
        return histogramMode(arr, 10);
    }

    private static double binaryStatsMeanLongstretch1SB(double[] arr){
        int[] meanBinary = new int[arr.length];
        double mean = mean(arr);
        for (int i = 0; i < arr.length; i++) if (arr[i] - mean > 0) meanBinary[i] = 1;

        double stretch = 0;
        double maxStretch = 0;
        for (int val : meanBinary){
            if (val == 1){
                stretch++;
            }
            else if (stretch > 0){
                if (stretch > maxStretch){
                    maxStretch = stretch;
                }
                stretch = 0;
            }
        }

        return maxStretch;
    }

    private static double outlierIncludeP001mdrmdDN(double[] arr){
        return outlierInclude(arr, true);
    }

    private static double outlierIncludeN001mdrmdDN(double[] arr){
        return outlierInclude(arr, false);
    }

    private static double histogramMode(double[] arr, int numBins){
        double min = min(arr);
        double max = max(arr);
        double binWidth = (max - min)/numBins;

        double[] histogram = new double[numBins];
        for (double val : arr) {
            int idx = (int) ((val - min) / binWidth);
            if (idx >= numBins) idx = numBins - 1;
            histogram[idx]++;
        }

        double[] centers = new double[numBins];
        for (int i = 0; i < numBins; i++){
            centers[i] = (min+(i*binWidth) + min+((i+1)*binWidth))/2;
        }

        int[] maxIndicies = argMax(histogram);
        double maxSum = 0;
        for (int idx : maxIndicies){
            maxSum += centers[idx];
        }

        return maxSum/maxIndicies.length;
    }

    private static double outlierInclude(double[] arr, boolean positive){
        double total = 0;
        double threshold = 0;

        for (double val : arr){
            if (!positive) val = -val;

            if (val >= 0){
                total++;
                if (val > threshold) threshold = val;
            }
        }

        int numThresholds = (int)(threshold/0.01)+1;
        double[] means = new double[numThresholds];
        double[] dists = new double[numThresholds];
        double[] medians = new double[numThresholds];
        for (int i = 0; i < numThresholds; i += 1){
            double d = i*0.01;

            ArrayList<Double> r = new ArrayList();
            if (positive){
                for (int n = 0; n < arr.length; n++) if (arr[n] >= d) r.add((double)n+1);
            }
            else{
                for (int n = 0; n < arr.length; n++) if (arr[n] <= -d) r.add((double)n+1);
            }

            double[] diff = new double[r.size()-1];
            for (int n = 0; n < diff.length; n++){
                diff[n] = r.get(n+1) - r.get(n);
            }
            int idx = (int)(d*100);

            means[idx] = mean(diff);
            dists[idx] = diff.length/total*100;


            medians[idx] = median(r.toArray(new Double[0]))/(arr.length/2)-1;
        }

        int firstNanIdx = -1;
        for (int i = 0; i < means.length; i++){
            if (Double.isNaN(means[i])){
                firstNanIdx = i;
                break;
            }
        }

        if (firstNanIdx != -1){
            dists = Arrays.copyOf(dists, firstNanIdx);
            medians = Arrays.copyOf(medians, firstNanIdx);
        }

        int thresholdIdx = -1;
        for (int i = dists.length-1; i >= 0; i--){
            if (dists[i] > 2){
                thresholdIdx = i;
                break;
            }
        }

        if (thresholdIdx != -1){
            medians = Arrays.copyOf(medians, thresholdIdx+1);
        }

        return median(medians);
    }

    public static void main(String[] args) throws Exception {
        int fold = 0;

        //Minimum working example
        String dataset = "FordA";
        Instances train = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\"+dataset+"\\"+dataset+"_TRAIN.arff");
        Instances test = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\"+dataset+"\\"+dataset+"_TEST.arff");
        Instances[] data = resampleTrainAndTestInstances(train, test, fold);
        train = data[0];
        test = data[1];

//        Catch22 c;
//        double accuracy;
//
//        c = new Catch22();
//        c.buildClassifier(train);
//        accuracy = ClassifierTools.accuracy(test, c);
//
//        System.out.println("Catch22 accuracy on " + dataset + " fold " + fold + " = " + accuracy);

        double[] inst = extractTimeSeries(train.get(0));
        System.out.println(Arrays.toString(inst));

        System.out.println(histMode5DN(inst));
        System.out.println(histMode10DN(inst));
        System.out.println(binaryStatsMeanLongstretch1SB(inst));
        System.out.println(outlierIncludeP001mdrmdDN(inst));
        System.out.println(outlierIncludeN001mdrmdDN(inst));
    }
}
