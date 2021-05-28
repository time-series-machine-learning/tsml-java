/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */

package tsml.transformers;

import experiments.data.DatasetLists;
import experiments.data.DatasetLoading;
import org.apache.commons.lang3.ArrayUtils;
import tsml.classifiers.shapelet_based.ShapeletTransformClassifier;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.TimeSeriesSummaryStatistics;
import utilities.ClassifierTools;
import utilities.generic_storage.Triple;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * Interface to determine how to resize the data.
 */
interface IResizeMetric {
    int calculateResizeValue(Map<Integer, Integer> counts);
}

/**
 * Interface to determine how to pad the data.
 */
interface IPadMetric {
    double calculatePadValue(double[] data);
}

/**
 * Class to resize data.
 */
public class Resizer implements TrainableTransformer {
    private int resizeLength;
    private boolean isFit = false;

    private IResizeMetric resizeMetric = new MedianResizeMetric();
    private IPadMetric padMetric = new MeanPadMetric();

    /**
     * Default constructor.
     */
    public Resizer() {}

    /**
     * Constructor to pass in the resizing metric.
     *
     * @param resizeMetric metric for resizing the data
     */
    public Resizer(IResizeMetric resizeMetric) {
        this.resizeMetric = resizeMetric;
    }

    /**
     * Constructor to pass in the padding metric.
     *
     * @param padMetric metric for padding the data
     */
    public Resizer(IPadMetric padMetric) {
        this.padMetric = padMetric;
    }

    /**
     * Constructor to pass in the resizing and padding metrics.
     *
     * @param resizeMetric metric for resizing the data
     * @param padMetric metric for padding the data
     */
    public Resizer(IResizeMetric resizeMetric, IPadMetric padMetric) {
        this(resizeMetric);
        this.padMetric = padMetric;
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
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        double[][] out = new double[inst.getNumDimensions()][resizeLength];

        int i = 0;
        for (TimeSeries ts : inst) {
            int diff = resizeLength - ts.getSeriesLength();
            double[] data = ts.toValueArray();

            // just need to copy data across, if we're the same or longer. truncate the
            // first values.
            if (diff <= 0) {
                System.arraycopy(data, 0, out[i], 0, resizeLength);
            }
            // we're shorter than the average
            else {
                System.arraycopy(data, 0, out[i], 0, data.length);
                for (int j = data.length; j < resizeLength; j++)
                    out[i][j] = padMetric.calculatePadValue(data);
            }

            //check if any NaNs exist as we want to overwrite those too.
            for (int j = 0; j < out[i].length; j++)
                if (Double.isNaN(out[i][j])) {
                    out[i][j] = padMetric.calculatePadValue(data);
                }

            //System.out.println(Arrays.toString(out[i]));

            i++;
        }

        return new TimeSeriesInstance(out, inst.getLabelIndex());
    }

    @Override
    public void fit(TimeSeriesInstances data) {
        Map<Integer, Integer> lengthCounts = data.getHistogramOfLengths();
        resizeLength = resizeMetric.calculateResizeValue(lengthCounts);
        isFit = true;
    }

    private Map<Integer, Integer> calculateLengthHistogram(Instances data) {
        Map<Integer, Integer> counts = new TreeMap<>();

        // build histogram of series lengths
        if (data.attribute(0).isRelationValued()) { // Multivariate
            for (Instance ins : data) {
                histogramLengths(ins.relationalValue(0), false, counts);
            }
        }
        else {
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

    @Override
    public void fit(Instances data) {
        // TODO Auto-generated method stub
    }

    /*
     * Example
     */
    public static void main(String[] args) throws Exception {
        //String local_path = "D:\\Work\\Data\\Multivariate_ts\\"; // Aarons local path for testing.
        String local_path = "Z:\\ArchiveData\\Univariate_arff\\";

        String output_dir = "Z:\\Personal Spaces\\Aaron's - safe space\\UnivariateUnequalPaddedProblems\\";

        //String dataset_name = "PLAID";
        for (String dataset_name : DatasetLists.variableLengthUnivariate) {
            System.out.println(dataset_name);
            TimeSeriesInstances train = DatasetLoading
                    .loadTSData(local_path + dataset_name + File.separator + dataset_name + "_TRAIN.arff");
            TimeSeriesInstances test = DatasetLoading
                    .loadTSData(local_path + dataset_name + File.separator + dataset_name + "_TEST.arff");

            if (train.hasMissing() || !train.isEqualLength()) {
                Resizer rs = new Resizer(new Resizer.MaxResizeMetric(), new Resizer.MeanNoisePadMetric());

                TimeSeriesInstances transformed_train = rs.fitTransform(train);
                TimeSeriesInstances transformed_test = rs.transform(test);

                DatasetLoading.saveTSDataset(transformed_train, output_dir + dataset_name + "\\" + dataset_name + "_TRAIN");
                DatasetLoading.saveTSDataset(transformed_test, output_dir + dataset_name + "\\" + dataset_name + "_TEST");
            }
        }
    }

    public static void test1() throws Exception {
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

    /*
     * RESIZE METRICS
     */

    /**
     * Static nested class to set the new size of each series to the weighted
     * median.
     */
    public static class WeightedMedianResizeMetric implements IResizeMetric {

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

    /**
     * Static nested class to set the new size of each series to the median.
     */
    public static class MedianResizeMetric implements IResizeMetric {

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

    /**
     * Static nested class to set the new size of each series to the maximum
     * length of a series.
     */
    public static class MaxResizeMetric implements IResizeMetric {

        @Override
        public int calculateResizeValue(Map<Integer, Integer> counts) {
            return counts.keySet().stream().max(Integer::compareTo).get();
        }
    }


    /*
     * PAD METRICS
     */

    /**
     * Static nested class to set the padding to the mean of a series.
     */
    public static class MeanPadMetric implements IPadMetric {

        @Override
        public double calculatePadValue(double[] data) {
            return TimeSeriesSummaryStatistics.mean(data);
        }
    }

    /**
     * Static nested class to set the padding to the mean of a series with some
     * noise added.
     */
    public static class MeanNoisePadMetric implements IPadMetric {
        Map<Integer, Triple<Double, Double, Double>> cache = new HashMap<>();
        Random random = new Random();

        @Override
        public double calculatePadValue(double[] data) {
            int hash = data.hashCode();

            Triple<Double, Double, Double> stats = cache.get(hash);
            if (stats == null) {
                stats = new Triple<Double, Double, Double>(TimeSeriesSummaryStatistics.max(data), TimeSeriesSummaryStatistics.min(data),
                        TimeSeriesSummaryStatistics.mean(Arrays.asList(ArrayUtils.toObject(data))));
                cache.put(hash, stats);
            }

            double scaledMax = stats.var1 / 100.0;
            double scaledMin = stats.var2 / 100.0;

            //mean + some noise between scaled min and max.
            return stats.var3 + (random.nextDouble() * (scaledMax - scaledMin)) + scaledMin;
        }
    }

    /**
     * Static nested class to set the padding to 0 or the value passed.
     */
    public static class FlatPadMetric implements IPadMetric {
        private final int padValue;

        public FlatPadMetric() {
            this.padValue = 0;
        }

        public FlatPadMetric(int value) {
            this.padValue = value;
        }

        @Override
        public double calculatePadValue(double[] data) {
            return padValue;
        }
    }
}
