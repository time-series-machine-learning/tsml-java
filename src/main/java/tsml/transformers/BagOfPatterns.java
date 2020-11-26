/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package tsml.transformers;

import experiments.data.DatasetLoading;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import utilities.NumUtils;
import utilities.StatisticalUtilities;

import java.io.File;
import java.util.*;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.SimpleBatchFilter;

/**
 * Filter to transform time series into a bag of patterns representation. i.e
 * pass a sliding window over each series normalise and convert each window to
 * sax form build a histogram of non-trivially matching patterns
 * 
 * Resulting in a bag (histogram) of patterns (SAX words) describing the
 * high-level structure of each timeseries
 *
 * Params: wordLength, alphabetSize, windowLength
 * 
 * @author James
 */
public class BagOfPatterns implements TrainableTransformer {

    public TreeSet<String> dictionary;

    private final int windowSize;
    private final int numIntervals;
    private final int alphabetSize;
    private boolean useRealAttributes = true;

    private boolean numerosityReduction = false; // can expand to different types of nr
    // like those in senin implementation later, if wanted

    private List<String> alphabet = null;

    private boolean isFit;

    private static final long serialVersionUID = 1L;

    public BagOfPatterns() {
        this(4, 4, 10);
    }

    public BagOfPatterns(int PAA_intervalsPerWindow, int SAX_alphabetSize, int windowSize) {
        this.numIntervals = PAA_intervalsPerWindow;
        this.alphabetSize = SAX_alphabetSize;
        this.windowSize = windowSize;
        alphabet = SAX.getAlphabet(SAX_alphabetSize);

    }

    public int getWindowSize() {
        return numIntervals;
    }

    public int getNumIntervals() {
        return numIntervals;
    }

    public int getAlphabetSize() {
        return alphabetSize;
    }

    public void useRealValuedAttributes(boolean b) {
        useRealAttributes = b;
    }

    public void performNumerosityReduction(boolean b) {
        numerosityReduction = b;
    }

    private HashMap<String, Integer> buildHistogram(LinkedList<double[]> patterns) {

        HashMap<String, Integer> hist = new HashMap<>();

        for (double[] pattern : patterns) {
            // convert to string
            String word = "";
            for (int j = 0; j < pattern.length; ++j)
                word += (String) alphabet.get((int) pattern[j]);

            Integer val = hist.get(word);
            if (val == null)
                val = 0;

            hist.put(word, val + 1);
        }

        return hist;
    }

    public HashMap<String, Integer> buildBag(TimeSeries series) {

        LinkedList<double[]> patterns = new LinkedList<>();

        double[] prevPattern = new double[windowSize];
        for (int i = 0; i < windowSize; ++i)
            prevPattern[i] = -1;

        for (int windowStart = 0; windowStart + windowSize - 1 < series.getSeriesLength(); ++windowStart) {
            double[] pattern = series.getSlidingWindowArray(windowStart, windowStart+windowSize);

            StatisticalUtilities.normInPlace(pattern);
            pattern = SAX.convertSequence(pattern, alphabetSize, numIntervals);

            if (!(numerosityReduction && identicalPattern(pattern, prevPattern)))
                patterns.add(pattern);
            prevPattern = pattern;
        }

        return buildHistogram(patterns);
    }


    public HashMap<String, Integer> buildBag(Instance series) {

        LinkedList<double[]> patterns = new LinkedList<>();

        double[] prevPattern = new double[windowSize];
        for (int i = 0; i < windowSize; ++i)
            prevPattern[i] = -1;

        for (int windowStart = 0; windowStart + windowSize - 1 < series.numAttributes() - 1; ++windowStart) {
            double[] pattern = slidingWindow(series, windowStart);

            StatisticalUtilities.normInPlace(pattern);
            pattern = SAX.convertSequence(pattern, alphabetSize, numIntervals);

            if (!(numerosityReduction && identicalPattern(pattern, prevPattern)))
                patterns.add(pattern);
            prevPattern = pattern;
        }

        return buildHistogram(patterns);
    }

    private double[] slidingWindow(Instance series, int windowStart) {
        double[] window = new double[windowSize];

        // copy the elements windowStart to windowStart+windowSize from data into the
        // window
        for (int i = 0; i < windowSize; ++i)
            window[i] = series.value(i + windowStart);

        return window;
    }

    private boolean identicalPattern(double[] a, double[] b) {
        for (int i = 0; i < a.length; ++i)
            if (a[i] != b[i])
                return false;

        return true;
    }

    public Instances determineOutputFormat(Instances inputFormat) {
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (String word : dictionary)
            attributes.add(new Attribute(word));

        Instances result = new Instances("BagOfPatterns_" + inputFormat.relationName(), attributes,
                inputFormat.numInstances());

        if (inputFormat.classIndex() >= 0) { // Classification set, set class
            // Get the class values as a fast vector
            Attribute target = inputFormat.attribute(inputFormat.classIndex());

            ArrayList<String> vals = new ArrayList<>(target.numValues());
            for (int i = 0; i < target.numValues(); i++) {
                vals.add(target.value(i));
            }

            result.insertAttributeAt(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals),
                    result.numAttributes());
            result.setClassIndex(result.numAttributes() - 1);
        }

        return result;
    }

    // TODO: Review, as we build bag twice on train. *Could* override fittransform
    // to avoid too much work.
    @Override
    public void fit(Instances data) {
        dictionary = new TreeSet<>();
        for (Instance inst : data) {
            HashMap<String, Integer> bag = buildBag(inst);
            dictionary.addAll(bag.keySet());
        }
        isFit = true;
    }

    @Override
    public void fit(TimeSeriesInstances data) {
        dictionary = new TreeSet<>();
        for (TimeSeriesInstance inst : data) {
            for (TimeSeries ts : inst){
                HashMap<String, Integer> bag = buildBag(ts);
                dictionary.addAll(bag.keySet());
            }
        }
        isFit = true;
    }

    @Override
    public boolean isFit() {
        return isFit;
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {

        //could do this across all dimensions.
        double[][] out = new double[inst.getNumDimensions()][];
        int i = 0;
        for(TimeSeries ts : inst){
            out[i++] = bagToArray(buildBag(ts));
        }
        
        //create a new output instance with the ACF data.
        return new TimeSeriesInstance(out, inst.getLabelIndex(), inst.getClassLabels());
    }

    @Override
    public Instance transform(Instance inst) {
        double[] bag = bagToArray(buildBag(inst));
        int size = bag.length + (inst.classIndex() >= 0 ? 1 : 0);
        Instance out = new DenseInstance(size);
        for (int j = 0; j < bag.length; j++)
            out.setValue(j, bag[j]);

        if (inst.classIndex() >= 0)
            out.setValue(out.numAttributes() - 1, inst.classValue());

        return out;
    }

    public double[] bagToArray(HashMap<String, Integer> bag) {
        double[] res = new double[dictionary.size()];

        int j = 0;
        for (String word : dictionary) {
            Integer val = bag.get(word);
            if (val != null)
                res[j] += val;
            ++j;
        }

        return res;
    }

    public static void main(String[] args) {
        String local_path = "D:\\Work\\Data\\Univariate_ts\\"; // Aarons local path for testing.
        String dataset_name = "Car";

        Instances train = DatasetLoading
                .loadData(local_path + dataset_name + File.separator + dataset_name + "_TRAIN.ts");
        Instances test = DatasetLoading
                .loadData(local_path + dataset_name + File.separator + dataset_name + "_TEST.ts");
        BagOfPatterns transform = new BagOfPatterns();
        Instances out_train = transform.fitTransform(train);
        Instances out_test = transform.transform(test);
        System.out.println(out_train.toString());
        System.out.println(out_test.toString());

    }





}
