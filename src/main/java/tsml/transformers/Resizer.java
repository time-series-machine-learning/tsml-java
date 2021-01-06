package tsml.transformers;

import java.io.File;
import java.util.ArrayList;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.TimeUnit;

import experiments.data.DatasetLoading;
import tsml.classifiers.shapelet_based.ShapeletTransformClassifier;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.TimeSeriesSummaryStatistics;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

interface IResizeMetric {
    public int calculateResizeValue(Map<Integer, Integer> counts);
}

interface IPadMetric{
    public double calculatePadValue(double[] data);
}

class WeightedMedianResizeMetric implements IResizeMetric {

    @Override
    public int calculateResizeValue(Map<Integer, Integer> counts) {
        int total_counts = counts.values().stream().mapToInt(e -> e.intValue()).sum();

        double cumulative_weight = 0;
        int median = 0;
        for (Map.Entry<Integer, Integer> entry : counts.entrySet()) {
            cumulative_weight += (double) entry.getValue() / (double) total_counts;
            median = entry.getKey();

            if (cumulative_weight > 1 / 2)
                break;
        }

        return median;
    }

}

class MedianResizeMetric implements IResizeMetric {

    @Override
    public int calculateResizeValue(Map<Integer, Integer> counts) {
        int total_counts = counts.values().stream().mapToInt(e -> e.intValue()).sum();

        // construct ordered list counts of keys.
        int[] keys = new int[total_counts];
        // {6 : 3, 10 : 1} => [6,6,6,10]
        int k = 0;
        for (Map.Entry<Integer, Integer> entry : counts.entrySet()) {
            for (int i = 0; i < entry.getValue(); i++)
                keys[k++] = entry.getKey();
        }

        int middle = keys.length / 2;
        if (keys.length % 2 == 1)
            return keys[middle];
        else
            return (keys[middle - 1] + keys[middle]) / 2;
    }

}


class MeanPadMetric implements IPadMetric{

    @Override
    public double calculatePadValue(double[] data) {
        return TimeSeriesSummaryStatistics.mean(data);
    }

}

public class Resizer implements TrainableTransformer {

    Map<Integer, Integer> lengthCounts;
    int resizeLength;
    boolean isFit = false;

    IResizeMetric lengthMetric = new MedianResizeMetric();
    IPadMetric padMetric = new MeanPadMetric();

    public Resizer() {}

    public Resizer(IResizeMetric length) {
        lengthMetric = length;
    }

    public Resizer(IPadMetric pad){
        padMetric = pad;
    }

    public Resizer(IResizeMetric length, IPadMetric pad){
        this(length);
        padMetric = pad;
    }



    @Override
    public Instance transform(Instance inst) {

        if (inst.attribute(0).isRelationValued()) { // Multivariate
            /*
             * for(Instance ins:data){
             * 
             * }
             */
            System.out.println("not implented multivariate yet");
            return null;
        } else {

            int length = Truncator.findLength(inst, true);

            int diff = resizeLength - length;

            double[] data = InstanceTools.ConvertInstanceToArrayRemovingClassValue(inst);
            double[] output = new double[resizeLength];

            // just need to copy data across, if we're the same or longer. truncate the
            // first values.
            if (diff <= 0) {
                System.arraycopy(data, 0, output, 0, resizeLength);
            }
            // we're shorter than the average
            else {
                // pad with mean.
                double pad = padMetric.calculatePadValue(data);

                System.arraycopy(data, 0, output, 0, length);
                for (int i = length; i < resizeLength; i++)
                    output[i] = pad;
            }

            DenseInstance out = new DenseInstance(resizeLength + 1);
            for (int i = 0; i < resizeLength; i++) {
                out.setValue(i, output[i]);
            }
            out.setValue(resizeLength, inst.classValue());

            return out;
        }
    }

    @Override
    public Instances determineOutputFormat(Instances inputFormat) {
        ArrayList<Attribute> atts = new ArrayList<>();
        Attribute a;
        for (int i = 0; i < resizeLength; i++) {
            a = new Attribute("Resizer" + (i + 1));
            atts.add(a);
        }
        if (inputFormat.classIndex() >= 0) { // Classification set, set class
            // Get the class values as a fast vector
            Attribute target = inputFormat.attribute(inputFormat.classIndex());
            ArrayList<String> vals = new ArrayList<>(target.numValues());
            for (int i = 0; i < target.numValues(); i++)
                vals.add(target.value(i));
            atts.add(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));

        }
        Instances result = new Instances("Resizer" + inputFormat.relationName(), atts, inputFormat.numInstances());
        if (inputFormat.classIndex() >= 0)
            result.setClassIndex(result.numAttributes() - 1);
        return result;
    }

    @Override
    public void fit(Instances data) {
        lengthCounts = calculateLengthHistogram(data);

        resizeLength = lengthMetric.calculateResizeValue(lengthCounts);

        isFit = true;
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        double[][] out = new double[inst.getNumDimensions()][resizeLength];

        int i=0;
        for(TimeSeries ts : inst){
            int diff = resizeLength - ts.getSeriesLength(); 
            double[] data = ts.toValueArray();

            // just need to copy data across, if we're the same or longer. truncate the
            // first values.
            if (diff <= 0) {
                System.arraycopy(data, 0, out[i], 0, resizeLength);
            }
            // we're shorter than the average
            else {
                // pad with mean.
                double pad = padMetric.calculatePadValue(data);

                System.arraycopy(data, 0, out[i], 0, data.length);
                for (int j = data.length; j < resizeLength; j++)
                    out[i][j] = pad;
            }

            i++;
        }

        return new TimeSeriesInstance(out, inst.getLabelIndex(), inst.getClassLabels());
    }

    @Override
    public void fit(TimeSeriesInstances data) {
        lengthCounts = data.getHistogramOfLengths();

        resizeLength = lengthMetric.calculateResizeValue(lengthCounts);

        isFit = true;
    }

    private Map<Integer, Integer> calculateLengthHistogram(Instances data) {
        Map<Integer, Integer> counts = new TreeMap<>();

        // build histogram of series lengths
        if (data.attribute(0).isRelationValued()) { // Multivariate
            for (Instance ins : data) {
                histogramLengths(ins.relationalValue(0), false, counts);
            }
        } else {
            histogramLengths(data, true, counts);
        }

        return counts;
    }

    private void histogramLengths(Instances d, boolean classValue, Map<Integer, Integer> counts) {
        for (Instance internal : d) {
            counts.merge(Truncator.findLength(internal, classValue), 1, Integer::sum);
        }
    }

    @Override
    public boolean isFit() {
        return isFit;
    }

    public static void main(String[] args) throws Exception {
        String local_path = "D:\\Work\\Data\\Univariate_ts\\"; // Aarons local path for testing.
        String dataset_name = "PLAID";
        Instances train = DatasetLoading
                .loadData(local_path + dataset_name + File.separator + dataset_name + "_TRAIN.ts");
        Instances test = DatasetLoading
                .loadData(local_path + dataset_name + File.separator + dataset_name + "_TEST.ts");

        Resizer rs = new Resizer();

        Map<Integer, Integer> map = rs.calculateLengthHistogram(train);
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            System.out.println("Train: " + entry.getKey() + " " + entry.getValue());
        }

        Instances resized_train = rs.fitTransform(train);
        map = rs.calculateLengthHistogram(resized_train);
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            System.out.println("Rezised_Train " + entry.getKey() + " " + entry.getValue());
        }

        Instances resized_test = rs.transform(test);

        ShapeletTransformClassifier stc = new ShapeletTransformClassifier();
        stc.setTrainTimeLimit(TimeUnit.MINUTES, 5);
        stc.buildClassifier(resized_train);

        double acc = ClassifierTools.accuracy(resized_test, stc);

        System.out.println(acc);
    }


    
}
