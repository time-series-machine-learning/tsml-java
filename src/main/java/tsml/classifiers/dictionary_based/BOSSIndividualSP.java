package tsml.classifiers.dictionary_based;

import com.carrotsearch.hppc.*;
import com.carrotsearch.hppc.cursors.DoubleIntCursor;
import com.carrotsearch.hppc.cursors.IntCursor;
import com.carrotsearch.hppc.cursors.ObjectDoubleCursor;
import com.carrotsearch.hppc.cursors.ObjectIntCursor;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;
import evaluation.evaluators.CrossValidationEvaluator;
import tsml.classifiers.MultiThreadable;
import tsml.classifiers.dictionary_based.bitword.BitWordLong;
import utilities.generic_storage.ComparableKeyPair;
import utilities.generic_storage.ComparablePair;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.core.*;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static tsml.classifiers.dictionary_based.WEASEL.MFT.*;
import static utilities.Utilities.argMax;

/**
 * BOSS classifier to be used with known parameters, for boss with parameter search, use BOSSEnsemble.
 *
 * Current implementation of BitWord as of 07/11/2016 only supports alphabetsize of 4, which is the expected value
 * as defined in the paper
 *
 * Params: wordLength, alphabetSize, windowLength, normalise?
 *
 * @author James Large. Enhanced by original author Patrick Schaefer
 *
 * Implementation based on the algorithm described in getTechnicalInformation()
 */
public class BOSSIndividualSP extends AbstractClassifier implements Serializable, Comparable<BOSSIndividualSP>, MultiThreadable {

    public boolean bigrams;
    public boolean featureSelection;
    public int fsLimit;
    public boolean histogramIntersection;
    public boolean wekaClassifier;
    public boolean tuningK = false;
    public ArrayList<PriorityQueue<ComparableKeyPair<Double,Double>>> neighbours = new ArrayList<>();
    public int maxK = 35;
    public int bestK = 1;
    public int numClasses = -1;

    public boolean IGB = false;
    public boolean anova = false;
    int maxWordLength;
    int[] bestValues;
    public boolean newDFT = false;
    public boolean newMFT = false;

    //all sfa words found in original buildClassifier(), no numerosity reduction/shortening applied
    protected BitWordLong[/*instance*/][/*windowindex*/] SFAwords;

    //histograms of words of the current wordlength with numerosity reduction applied (if selected)
    protected ArrayList<SPBag> bags;

    //breakpoints to be found by MCB
    protected double[/*letterindex*/][/*breakpointsforletter*/] breakpoints;

    protected double inverseSqrtWindowSize;
    protected int windowSize;
    protected int wordLength;
    protected int alphabetSize;
    protected boolean norm;
    protected boolean numerosityReduction = true;
    protected boolean cleanAfterBuild = false;

    protected int levels;
    protected double levelWeighting = 0.5;
    protected int seriesLength;

    protected ObjectHashSet<ComparablePair<BitWordLong, Byte>> chiSquare;
    protected double chiLimit;

    protected double accuracy = -1;
    protected double weight = 1;
    protected ArrayList<Integer> subsampleIndices;
    protected ArrayList<Integer> trainPreds;

    protected boolean multiThread = false;
    protected int numThreads = 1;
    protected ExecutorService ex;

    protected int seed = 0;
    protected Random rand;

    protected static final long serialVersionUID = 22551L;



    Instances transformedData;
    Instances header;
    Classifier classifier;
    HashMap<ComparablePair<BitWordLong, Byte>, Integer> words;


    public BOSSIndividualSP(int wordLength, int alphabetSize, int windowSize, boolean normalise, int levels, double chiLimit, boolean multiThread, int numThreads, ExecutorService ex) {
        this.maxWordLength = wordLength;

        this.wordLength = wordLength;
        this.alphabetSize = alphabetSize;
        this.windowSize = windowSize;
        this.inverseSqrtWindowSize = 1.0 / Math.sqrt(windowSize);
        this.norm = normalise;
        this.levels = levels;
        this.chiLimit = chiLimit;
        this.multiThread = multiThread;
        this.numThreads = numThreads;
        this.ex = ex;
    }

    public BOSSIndividualSP(int wordLength, int alphabetSize, int windowSize, boolean normalise, int levels, double chiLimit) {
        this.wordLength = wordLength;
        this.alphabetSize = alphabetSize;
        this.windowSize = windowSize;
        this.inverseSqrtWindowSize = 1.0 / Math.sqrt(windowSize);
        this.norm = normalise;
        this.levels = levels;
        this.chiLimit = chiLimit;
    }

    /**
     * Used when shortening histograms, copies 'meta' data over, but with shorter
     * word length, actual shortening happens separately
     */
    public BOSSIndividualSP(BOSSIndividualSP boss, int wordLength) {
        this.wordLength = wordLength;

        this.windowSize = boss.windowSize;
        this.inverseSqrtWindowSize = boss.inverseSqrtWindowSize;
        this.alphabetSize = boss.alphabetSize;
        this.norm = boss.norm;
        this.numerosityReduction = boss.numerosityReduction;

        this.levels = boss.levels;
        this.levelWeighting = boss.levelWeighting;
        this.seriesLength = boss.seriesLength;

        this.chiLimit = boss.chiLimit;

        this.SFAwords = boss.SFAwords;
        this.breakpoints = boss.breakpoints;

        this.multiThread = boss.multiThread;
        this.numThreads = boss.numThreads;
        this.ex = boss.ex;

        this.seed = boss.seed;
        this.rand = boss.rand;

        this.bags = new ArrayList<>(boss.bags.size());
    }

    @Override
    public int compareTo(BOSSIndividualSP o) {
        return Double.compare(this.accuracy, o.accuracy);
    }

    @Override
    public void enableMultiThreading(int numThreads) {
        this.numThreads = numThreads;
    }

    //map of <word, level> => count
    public static class SPBag extends HashMap<ComparablePair<BitWordLong, Byte>, Integer> {
        double classVal;

        public SPBag() {
            super();
        }

        public SPBag(double classValue) {
            super();
            classVal = classValue;
        }

        public double getClassVal() { return classVal; }
        public void setClassVal(double classVal) { this.classVal = classVal; }
    }

    public int getWindowSize() { return windowSize; }
    public int getWordLength() { return wordLength; }
    public int getAlphabetSize() { return alphabetSize; }
    public boolean isNorm() { return norm; }

    public ArrayList<SPBag> getBags() { return bags; }

    /**
     * @return { numIntervals(word length), alphabetSize, slidingWindowSize, normalise? }
     */
    public int[] getParameters() {
        return new int[] { wordLength, alphabetSize, windowSize };
    }

    public void setSeed(int i){ seed = i; }

    public void clean() {
        SFAwords = null;

        if (wekaClassifier){
            bags = null;
        }
    }

    protected double[][] performDFT(double[][] windows) {
        double[][] dfts = new double[windows.length][wordLength];
        for (int i = 0; i < windows.length; ++i) {
            dfts[i] = DFT(windows[i]);
        }
        return dfts;
    }

    protected double stdDev(double[] series) {
        double sum = 0.0;
        double squareSum = 0.0;
        for (int i = 0; i < windowSize; i++) {
            sum += series[i];
            squareSum += series[i]*series[i];
        }

        double mean = sum / series.length;
        double variance = squareSum / series.length - mean*mean;
        return variance > 0 ? Math.sqrt(variance) : 1.0;
    }

    protected double[] DFT(double[] series) {
        if (anova || newDFT){
            double[] data = new double[this.windowSize];
            System.arraycopy(series, 0, data, 0, Math.min(this.windowSize, series.length));

            DoubleFFT_1D fft = new DoubleFFT_1D(this.windowSize);
            int startOffset =  norm ? 2 : 0;

            fft.realForward(data);
            data[1] = 0; // DC-coefficient imaginary part

            // make it even length for uneven windowSize
            double[] copy = new double[maxWordLength];
            int length = Math.min(this.windowSize - startOffset, maxWordLength);
            System.arraycopy(data, startOffset, copy, 0, length);

            // norming
            int sign = 1;
            for (int i = 0; i < copy.length; i++) {
                copy[i] *= sign;
                sign *= -1;
            }

            return copy;
        }

        //taken from FFT.java but
        //return just a double[] size n, { real1, imag1, ... realn/2, imagn/2 }
        //instead of Complex[] size n/2

        //only calculating first wordlength/2 coefficients (output values),
        //and skipping first coefficient if the data is to be normalised
        int n=series.length;
        int outputLength = wordLength/2;
        int start = (norm ? 1 : 0);

        //normalize the disjoint windows and sliding windows by dividing them by their standard deviation
        //all Fourier coefficients are divided by sqrt(windowSize)

        double normalisingFactor = inverseSqrtWindowSize / stdDev(series);

        double[] dft=new double[outputLength*2];

        for (int k = start; k < start + outputLength; k++) {  // For each output element
            float sumreal = 0;
            float sumimag = 0;
            for (int t = 0; t < n; t++) {  // For each input element
                sumreal +=  series[t]*Math.cos(2*Math.PI * t * k / n);
                sumimag += -series[t]*Math.sin(2*Math.PI * t * k / n);
            }
            dft[(k-start)*2]   = sumreal * normalisingFactor;
            dft[(k-start)*2+1] = sumimag * normalisingFactor;
        }
        return dft;
    }

    private double[] DFTunnormed(double[] series) {
        //taken from FFT.java but
        //return just a double[] size n, { real1, imag1, ... realn/2, imagn/2 }
        //instead of Complex[] size n/2

        //only calculating first wordlength/2 coefficients (output values),
        //and skipping first coefficient if the data is to be normalised
        int n=series.length;
        int outputLength = maxWordLength/2;
        int start = (norm ? 1 : 0);

        double[] dft = new double[outputLength*2];
        double twoPi = 2*Math.PI / n;

        for (int k = start; k < start + outputLength; k++) {  // For each output element
            float sumreal = 0;
            float sumimag = 0;
            for (int t = 0; t < n; t++) {  // For each input element
                sumreal +=  series[t]*Math.cos(twoPi * t * k);
                sumimag += -series[t]*Math.sin(twoPi * t * k);
            }
            dft[(k-start)*2]   = sumreal;
            dft[(k-start)*2+1] = sumimag;
        }
        return dft;
    }

    private double[] normalizeDFT(double[] dft, double std) {
        double normalisingFactor = (std > 0? 1.0 / std : 1.0) * inverseSqrtWindowSize;
        for (int i = 0; i < dft.length; i++)
            dft[i] *= normalisingFactor;

        return dft;
    }

    private double[][] performMFT(double[] series) {
        if (newMFT){
            DoubleFFT_1D fft = new DoubleFFT_1D(this.windowSize);
            int startOffset =  norm ? 2 : 0;

            int wordLength = Math.min(windowSize, maxWordLength + startOffset);
            wordLength += wordLength%2; // make it even
            double[] phis = new double[wordLength];

            for (int u = 0; u < phis.length; u += 2) {
                double uHalve = -u / 2;
                phis[u] = realPartEPhi(uHalve, this.windowSize);
                phis[u + 1] = complexPartEPhi(uHalve, this.windowSize);
            }

            // means and stddev for each sliding window
            int end = Math.max(1, series.length - this.windowSize + 1);
            double[] means = new double[end];
            double[] stds = new double[end];
            calcIncrementalMeanStddev(this.windowSize, series, means, stds);

            double[][] transformed = new double[end][];

            // holds the DFT of each sliding window
            double[] mftData = new double[wordLength];
            double[] data = series;

            for (int t = 0; t < end; t++) {
                // use the MFT
                if (t > 0) {
                    for (int k = 0; k < wordLength; k += 2) {
                        double real1 = (mftData[k] + data[t + this.windowSize - 1] - data[t - 1]);
                        double imag1 = (mftData[k + 1]);

                        double real = complexMultiplyRealPart(real1, imag1, phis[k], phis[k + 1]);
                        double imag = complexMultiplyImagPart(real1, imag1, phis[k], phis[k + 1]);

                        mftData[k] = real;
                        mftData[k + 1] = imag;
                    }
                }
                // use the DFT for the first offset
                else {
                    double[] dft = new double[this.windowSize];
                    double[] data2 = series;
                    System.arraycopy(data2, 0, dft, 0, Math.min(this.windowSize, data.length));

                    fft.realForward(dft);
                    dft[1] = 0; // DC-coefficient imag part

                    // if windowSize > mftData.queryLength, the remaining data should be 0 now.
                    System.arraycopy(dft, 0, mftData, 0, Math.min(mftData.length, dft.length));

//          double[] dft = new double[this.windowSize];
//          System.arraycopy(toArrayNoClass(timeSeries), 0, dft, 0, Math.min(this.windowSize, instanceLength(timeSeries)));
//          dft = transform(dft, dft.length, true);
//          dft[1] = 0; // DC-coefficient imag part
//
//          // if windowSize > mftData.queryLength, the remaining data should be 0 now.
//          System.arraycopy(dft, 0, mftData, 0, Math.min(mftData.length, dft.length));
                }

                double[] copy = new double[maxWordLength];
                System.arraycopy(mftData, startOffset, copy, 0, Math.min(maxWordLength, mftData.length-startOffset));

                transformed[t] = normalizeFT(copy, stds[t]);
            }

            return transformed;
        }

        // ignore DC value?
        int startOffset = norm ? 2 : 0;
        int l = maxWordLength;
        l = l + l % 2; // make it even
        double[] phis = new double[l];
        for (int u = 0; u < phis.length; u += 2) {
            double uHalve = -(u + startOffset) / 2;
            phis[u] = realephi(uHalve, windowSize);
            phis[u + 1] = complexephi(uHalve, windowSize);
        }

        // means and stddev for each sliding window
        int end = Math.max(1, series.length - windowSize + 1);
        double[] means = new double[end];
        double[] stds = new double[end];
        calcIncrementalMeanStddev(windowSize, series, means, stds);
        // holds the DFT of each sliding window
        double[][] transformed = new double[end][];
        double[] mftData = null;

        for (int t = 0; t < end; t++) {
            // use the MFT
            if (t > 0) {
                for (int k = 0; k < l; k += 2) {
                    double real1 = (mftData[k] + series[t + windowSize - 1] - series[t - 1]);
                    double imag1 = (mftData[k + 1]);
                    double real = complexMulReal(real1, imag1, phis[k], phis[k + 1]);
                    double imag = complexMulImag(real1, imag1, phis[k], phis[k + 1]);
                    mftData[k] = real;
                    mftData[k + 1] = imag;
                }
            } // use the DFT for the first offset
            else {
                mftData = Arrays.copyOf(series, windowSize);
                mftData = DFTunnormed(mftData);
            }
            // normalization for lower bounding
            transformed[t] = normalizeDFT(Arrays.copyOf(mftData, l), stds[t]);
        }
        return transformed;
    }

    private void calcIncrementalMeanStddev(int windowLength, double[] series, double[] means, double[] stds) {
        double sum = 0;
        double squareSum = 0;
        // it is faster to multiply than to divide
        double rWindowLength = 1.0 / (double) windowLength;
        double[] tsData = series;
        for (int ww = 0; ww < windowLength; ww++) {
            sum += tsData[ww];
            squareSum += tsData[ww] * tsData[ww];
        }
        means[0] = sum * rWindowLength;
        double buf = squareSum * rWindowLength - means[0] * means[0];
        stds[0] = buf > 0 ? Math.sqrt(buf) : 0;
        for (int w = 1, end = tsData.length - windowLength + 1; w < end; w++) {
            sum += tsData[w + windowLength - 1] - tsData[w - 1];
            means[w] = sum * rWindowLength;
            squareSum += tsData[w + windowLength - 1] * tsData[w + windowLength - 1] - tsData[w - 1] * tsData[w - 1];
            buf = squareSum * rWindowLength - means[w] * means[w];
            stds[w] = buf > 0 ? Math.sqrt(buf) : 0;
        }
    }

    private static double complexMulReal(double r1, double im1, double r2, double im2) {
        return r1 * r2 - im1 * im2;
    }

    private static double complexMulImag(double r1, double im1, double r2, double im2) {
        return r1 * im2 + r2 * im1;
    }

    private static double realephi(double u, double M) {
        return Math.cos(2 * Math.PI * u / M);
    }

    private static double complexephi(double u, double M) {
        return -Math.sin(2 * Math.PI * u / M);
    }

    protected double[][] disjointWindows(double [] data) {
        int amount = (int)Math.ceil(data.length/(double)windowSize);
        double[][] subSequences = new double[amount][windowSize];

        for (int win = 0; win < amount; ++win) {
            int offset = Math.min(win*windowSize, data.length-windowSize);

            //copy the elements windowStart to windowStart+windowSize from data into
            //the subsequence matrix at position windowStart
            System.arraycopy(data,offset,subSequences[win],0,windowSize);
        }

        return subSequences;
    }

    protected double[][] MCB(Instances data) {
        double[][][] dfts = new double[data.numInstances()][][];

        ArrayList<double[]> samples = new ArrayList<>();
        ArrayList<double[]> transformedSamples = new ArrayList<>();
        ArrayList<Double> labels = new ArrayList<>();

        int sample = 0;
        for (Instance inst : data) {
            double[][] windows = disjointWindows(toArrayNoClass(inst));
            dfts[sample] = performDFT(windows); //approximation

            for (int i = 0; i < dfts[sample].length; i++) {
                samples.add(windows[i]);
                transformedSamples.add(dfts[sample][i]);
                labels.add(inst.classValue());
            }

            sample++;
        }

        double[][] allSamples = new double[samples.size()][];
        double[][] allTransformedSamples = new double[transformedSamples.size()][];
        double[] allLabels = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            allSamples[i] = samples.get(i);
            allTransformedSamples[i] = transformedSamples.get(i);
            allLabels[i] = labels.get(i);
        }

        int numInsts = dfts.length;
        int numWindowsPerInst = dfts[0].length;
        int totalNumWindows = numInsts*numWindowsPerInst;

        breakpoints = new double[wordLength][alphabetSize];

        for (int letter = 0; letter < wordLength; ++letter) { //for each dft coeff

            //extract this column from all windows in all instances
            double[] column = new double[totalNumWindows];
            for (int inst = 0; inst < numInsts; ++inst)
                for (int window = 0; window < numWindowsPerInst; ++window) {
                    //rounding dft coefficients to reduce noise
                    column[(inst * numWindowsPerInst) + window] = Math.round(dfts[inst][window][letter]*100.0)/100.0;
                }

            //sort, and run through to find breakpoints for equi-depth bins
            Arrays.sort(column);

            double binIndex = 0;
            double targetBinDepth = (double)totalNumWindows / (double)alphabetSize;

            for (int bp = 0; bp < alphabetSize-1; ++bp) {
                binIndex += targetBinDepth;
                breakpoints[letter][bp] = column[(int)binIndex];
            }

            breakpoints[letter][alphabetSize-1] = Double.MAX_VALUE; //last one can always = infinity
        }

        if (anova){
            anova(allSamples, allTransformedSamples, allLabels);
        }

        return breakpoints;
    }

    ///// WEASEL TEST START /////

    protected double[][] IGB(Instances data) {
        ArrayList<ComparablePair<Double,Double>>[] orderline = new ArrayList[wordLength];
        for (int i = 0; i < orderline.length; i++) {
            orderline[i] = new ArrayList<>();
        }

        ArrayList<double[]> samples = new ArrayList<>();
        ArrayList<double[]> transformedSamples = new ArrayList<>();
        ArrayList<Double> labels = new ArrayList<>();
        for (Instance inst : data) {
            double[][] windows = disjointWindows(toArrayNoClass(inst));
            double[][] dfts = performDFT(windows); //approximation

            for (int i = 0; i < dfts.length; i++) {
                for (int n = 0; n < dfts[i].length; n++) {
                    // round to 2 decimal places to reduce noise
                    double value = Math.round(dfts[i][n] * 100.0) / 100.0;

                    orderline[n].add(new ComparablePair<>(value, inst.classValue()));
                }

                samples.add(windows[i]);
                transformedSamples.add(dfts[i]);
                labels.add(inst.classValue());
            }
        }

        double[][] allSamples = new double[samples.size()][];
        double[][] allTransformedSamples = new double[transformedSamples.size()][];
        double[] allLabels = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            allSamples[i] = samples.get(i);
            allTransformedSamples[i] = transformedSamples.get(i);
            allLabels[i] = labels.get(i);
        }

        breakpoints = new double[wordLength][alphabetSize];

        for (int i = 0; i < orderline.length; i++) {
            if (!orderline[i].isEmpty()) {
                Collections.sort(orderline[i]);

                ArrayList<Integer> splitPoints = new ArrayList<>();
                findBestSplit(orderline[i], 0, orderline[i].size(), alphabetSize, splitPoints);

                Collections.sort(splitPoints);

                for (int n = 0; n < splitPoints.size(); n++) {
                   breakpoints[i][n] = orderline[i].get(splitPoints.get(n) + 1).var1;
                }

                breakpoints[i][alphabetSize-1] = Double.MAX_VALUE;
            }
        }

        if (anova){
            anova(allSamples, allTransformedSamples, allLabels);
        }

        return breakpoints;
    }

    protected void findBestSplit(List<ComparablePair<Double,Double>> element, int start, int end, int remainingSymbols,
                                 List<Integer> splitPoints) {
        double bestGain = -1;
        int bestPos = -1;

        // class entropy
        int total = end - start;
        ObjectIntHashMap<Double> cIn = new ObjectIntHashMap<>();
        ObjectIntHashMap<Double> cOut = new ObjectIntHashMap<>();
        for (int pos = start; pos < end; pos++) {
            cOut.putOrAdd(element.get(pos).var2, 1, 1);
        }
        double class_entropy = entropy(cOut, total);

        int i = start;
        Double lastLabel = element.get(i).var2;
        i += moveElement(element, cIn, cOut, start);

        for (int split = start + 1; split < end - 1; split++) {
            Double label = element.get(i).var2;
            i += moveElement(element, cIn, cOut, split);

            // only inspect changes of the label
            if (!label.equals(lastLabel)) {
                double gain = calculateInformationGain(cIn, cOut, class_entropy, i, total);
                gain = Math.round(gain * 1000.0) / 1000.0; // round for 4 decimal places

                if (gain >= bestGain) {
                    bestPos = split;
                    bestGain = gain;
                }
            }
            lastLabel = label;
        }

        if (bestPos > -1) {
            splitPoints.add(bestPos);

            // recursive split
            remainingSymbols = remainingSymbols / 2;
            if (remainingSymbols > 1) {
                if (bestPos - start > 2 && end - bestPos > 2) { // enough data points?
                    findBestSplit(element, start, bestPos, remainingSymbols, splitPoints);
                    findBestSplit(element, bestPos, end, remainingSymbols, splitPoints);
                } else if (end - bestPos > 4) { // enough data points?
                    findBestSplit(element, bestPos, (end - bestPos) / 2, remainingSymbols, splitPoints);
                    findBestSplit(element, (end - bestPos) / 2, end, remainingSymbols, splitPoints);
                } else if (bestPos - start > 4) { // enough data points?
                    findBestSplit(element, start, (bestPos - start) / 2, remainingSymbols, splitPoints);
                    findBestSplit(element, (bestPos - start) / 2, end, remainingSymbols, splitPoints);
                }
            }
        }
    }

    protected double entropy(ObjectIntHashMap<Double> frequency, double total) {
        double entropy = 0;
        double log2 = 1.0 / Math.log(2.0);
        for (IntCursor element : frequency.values()) {
            double p = element.value / total;
            if (p > 0) {
                entropy -= p * Math.log(p) * log2;
            }
        }
        return entropy;
    }

    protected double calculateInformationGain(ObjectIntHashMap<Double> cIn, ObjectIntHashMap<Double> cOut,
                                              double class_entropy, double total_c_in, double total) {
        double total_c_out = (total - total_c_in);
        return class_entropy
                - total_c_in / total * entropy(cIn, total_c_in)
                - total_c_out / total * entropy(cOut, total_c_out);
    }

    protected int moveElement(List<ComparablePair<Double,Double>> element, ObjectIntHashMap<Double> cIn,
                              ObjectIntHashMap<Double> cOut, int pos) {
        cIn.putOrAdd(element.get(pos).var2, 1, 1);
        cOut.putOrAdd(element.get(pos).var2, -1, -1);
        return 1;
    }

    public void anova(double[][] samples, double[][] transformedSignal, double[] labels){
        WEASEL.SFASupervised.Indices<Double>[] best = calcBestCoefficients(samples, labels, transformedSignal);

        // use best coefficients (the ones with largest f-value)
        bestValues = new int[Math.min(best.length, wordLength)];
        maxWordLength = 0;
        for (int i = 0; i < this.bestValues.length; i++) {
            bestValues[i] = best[i].index;
            maxWordLength = Math.max(best[i].index + 1, maxWordLength);
        }

        // make sure it is an even number
        maxWordLength += maxWordLength % 2;

        System.out.println(wordLength + " " + maxWordLength + " " + bestValues.length);
    }


    /**
     * calculate ANOVA F-stat
     * compare : https://github.com/scikit-learn/scikit-learn/blob/c957249/sklearn/feature_selection/univariate_selection.py#L121
     *
     * @param transformedSignal
     * @return
     */
    public static WEASEL.SFASupervised.Indices<Double>[] calcBestCoefficients(
            double[][] samples,
            double[] labels,
            double[][] transformedSignal) {
        HashMap<Double, ArrayList<double[]>> classes = new HashMap<>();
        for (int i = 0; i < samples.length; i++) {
            ArrayList<double[]> allTs = classes.get(labels[i]);
            if (allTs == null) {
                allTs = new ArrayList<>();
                classes.put(labels[i], allTs);
            }
            allTs.add(transformedSignal[i]);
        }

        double nSamples = transformedSignal.length;
        double nClasses = classes.keySet().size();

//    int length = 0;
//    for (int i = 0; i < transformedSignal.length; i++) {
//      length = Math.max(transformedSignal[i].length, length);
//    }
        int length = (transformedSignal != null && transformedSignal.length > 0) ? transformedSignal[0].length : 0;

        double[] f = getFoneway(length, classes, nSamples, nClasses);

        // sort by largest f-value
        @SuppressWarnings("unchecked")
        List<WEASEL.SFASupervised.Indices<Double>> best = new ArrayList<>(f.length);
        for (int i = 0; i < f.length; i++) {
            if (!Double.isNaN(f[i])) {
                best.add(new WEASEL.SFASupervised.Indices<>(i, f[i]));
            }
        }
        Collections.sort(best);
        return best.toArray(new WEASEL.SFASupervised.Indices[]{});
    }

    /**
     * The one-way ANOVA tests the null hypothesis that 2 or more groups have
     * the same population mean. The test is applied to samples from two or
     * more groups, possibly with differing sizes.
     *
     * @param length
     * @param classes
     * @param nSamples
     * @param nClasses
     * @return
     */
    public static double[] getFoneway(
            int length,
            Map<Double, ArrayList<double[]>> classes,
            double nSamples,
            double nClasses) {
        double[] ss_alldata = new double[length];
        HashMap<Double, double[]> sums_args = new HashMap<>();

        for (Map.Entry<Double, ArrayList<double[]>> allTs : classes.entrySet()) {

            double[] sums = new double[ss_alldata.length];
            sums_args.put(allTs.getKey(), sums);

            for (double[] ts : allTs.getValue()) {
                for (int i = 0; i < ts.length; i++) {
                    ss_alldata[i] += ts[i] * ts[i];
                    sums[i] += ts[i];
                }
            }
        }

        double[] square_of_sums_alldata = new double[ss_alldata.length];
        Map<Double, double[]> square_of_sums_args = new HashMap<>();
        for (Map.Entry<Double, double[]> sums : sums_args.entrySet()) {
            for (int i = 0; i < sums.getValue().length; i++) {
                square_of_sums_alldata[i] += sums.getValue()[i];
            }

            double[] squares = new double[sums.getValue().length];
            square_of_sums_args.put(sums.getKey(), squares);
            for (int i = 0; i < sums.getValue().length; i++) {
                squares[i] += sums.getValue()[i] * sums.getValue()[i];
            }
        }

        for (int i = 0; i < square_of_sums_alldata.length; i++) {
            square_of_sums_alldata[i] *= square_of_sums_alldata[i];
        }

        double[] sstot = new double[ss_alldata.length];
        for (int i = 0; i < sstot.length; i++) {
            sstot[i] = ss_alldata[i] - square_of_sums_alldata[i] / nSamples;
        }

        double[] ssbn = new double[ss_alldata.length];    // sum of squares between
        double[] sswn = new double[ss_alldata.length];    // sum of squares within

        for (Map.Entry<Double, double[]> sums : square_of_sums_args.entrySet()) {
            double n_samples_per_class = classes.get(sums.getKey()).size();
            for (int i = 0; i < sums.getValue().length; i++) {
                ssbn[i] += sums.getValue()[i] / n_samples_per_class;
            }
        }

        for (int i = 0; i < square_of_sums_alldata.length; i++) {
            ssbn[i] -= square_of_sums_alldata[i] / nSamples;
        }

        double dfbn = nClasses - 1;                       // degrees of freedom between
        double dfwn = nSamples - nClasses;              // degrees of freedom within
        double[] msb = new double[ss_alldata.length];   // variance (mean square) between classes
        double[] msw = new double[ss_alldata.length];   // variance (mean square) within samples
        double[] f = new double[ss_alldata.length];     // f-ratio

        for (int i = 0; i < sswn.length; i++) {
            sswn[i] = sstot[i] - ssbn[i];
            msb[i] = ssbn[i] / dfbn;
            msw[i] = sswn[i] / dfwn;
            f[i] = msb[i] / msw[i];
        }
        return f;
    }

    protected BitWordLong createWordAnova(double[] dft) {
        BitWordLong word = new BitWordLong(wordLength);
        int length = Math.min(wordLength, this.bestValues.length);
        for (int i = 0; i < length; ++i) { //for each letter
            int l = bestValues[i];
            for (int bp = 0; bp < alphabetSize; ++bp) //run through breakpoints until right one found
                if (dft[l] <= breakpoints[l][bp]) {
                    word.push(bp); //add corresponding letter to word
                    break;
                }
        }

        return word;
    }

    ///// WEASEL TEST END /////

    /**
     * Builds a brand new boss bag from the passed fourier transformed data, rather than from
     * looking up existing transforms from earlier builds (i.e. SFAWords).
     *
     * to be used e.g to transform new test instances
     */
    protected SPBag createSPBagSingle(double[][] dfts) {
        SPBag bag = new SPBag();
        BitWordLong lastWord = new BitWordLong();
        BitWordLong[] words = new BitWordLong[dfts.length];

        int wInd = 0;
        int trivialMatchCount = 0;

        for (double[] d : dfts) {
            BitWordLong word;
            if (anova) word = createWordAnova(d);
            else word = createWord(d);

            words[wInd] = word;

            if (bigrams) {
                if (wInd - windowSize >= 0 && lastWord.getWord() != 0) {
                    BitWordLong bigram = new BitWordLong(words[wInd - windowSize], word);

                    ComparablePair<BitWordLong, Byte> key = new ComparablePair<>(bigram, (byte) 0);
                    Integer val = bag.get(key);

                    if (val == null)
                        val = 0;
                    bag.put(key, ++val);
                }
            }

            //add to bag, unless num reduction applies
            if (numerosityReduction && word.equals(lastWord)) {
                ++trivialMatchCount;
                ++wInd;
            }
            else {
                //if a run of equivalent words, those words essentially representing the same
                //elongated pattern. still apply numerosity reduction, however use the central
                //time position of the elongated pattern to represent its position
                addWordToPyramid(word, wInd - (trivialMatchCount/2), bag);

                lastWord = word;
                trivialMatchCount = 0;
                ++wInd;
            }
        }

        applyPyramidWeights(bag);

        return bag;
    }

    protected BitWordLong createWord(double[] dft) {
        BitWordLong word = new BitWordLong(wordLength);
        for (int l = 0; l < wordLength; ++l) //for each letter
            for (int bp = 0; bp < alphabetSize; ++bp) //run through breakpoints until right one found
                if (dft[l] <= breakpoints[l][bp]) {
                    word.push(bp); //add corresponding letter to word
                    break;
                }

        return word;
    }

    /**
     * @return data of passed instance in a double array with the class value removed if present
     */
    protected static double[] toArrayNoClass(Instance inst) {
        int length = inst.numAttributes();
        if (inst.classIndex() >= 0)
            --length;

        double[] data = new double[length];

        for (int i=0, j=0; i < inst.numAttributes(); ++i)
            if (inst.classIndex() != i)
                data[j++] = inst.value(i);

        return data;
    }

    /**
     * @return BOSSSpatialPyramidsTransform-ed bag, built using current parameters
     */
    public SPBag BOSSSpatialPyramidsTransform(Instance inst) {

        double[][] mfts = performMFT(toArrayNoClass(inst)); //approximation
        SPBag bag2 = createSPBagSingle(mfts); //discretisation/bagging
        bag2.setClassVal(inst.classValue());

        return bag2;
    }

    /**
     * Shortens all bags in this BOSSSpatialPyramids_Redo instance (histograms) to the newWordLength, if wordlengths
     * are same, instance is UNCHANGED
     *
     * @param newWordLength wordLength to shorten it to
     * @return new boss classifier with newWordLength, or passed in classifier if wordlengths are same
     */
    public BOSSIndividualSP buildShortenedSPBags(int newWordLength) throws Exception {
        if (newWordLength == wordLength) //case of first iteration of word length search in ensemble
            return this;
        if (newWordLength > wordLength)
            throw new Exception("Cannot incrementally INCREASE word length, current:"+wordLength+", requested:"+newWordLength);
        if (newWordLength < 2)
            throw new Exception("Invalid wordlength requested, current:"+wordLength+", requested:"+newWordLength);

        BOSSIndividualSP newBoss = new BOSSIndividualSP(this, newWordLength);

        //build hists with new word length from SFA words, and copy over the class values of original insts
        for (int i = 0; i < bags.size(); ++i) {
            SPBag newSPBag = createSPBagFromWords(newWordLength, SFAwords[i]);
            newSPBag.setClassVal(bags.get(i).getClassVal());
            newBoss.bags.add(newSPBag);
        }

        return newBoss;
    }

    /**
     * Builds a bag from the set of words for a pre-transformed series of a given wordlength.
     */
    protected SPBag createSPBagFromWords(int thisWordLength, BitWordLong[] words) {
        SPBag bag = new SPBag();
        BitWordLong lastWord = new BitWordLong();
        BitWordLong[] newWords = new BitWordLong[words.length];

        int wInd = 0;
        int trivialMatchCount = 0; //keeps track of how many words have been the same so far

        for (BitWordLong w : words) {
            BitWordLong word = new BitWordLong(w);
            if (wordLength != thisWordLength)
                word.shorten(16-thisWordLength); //max word length, no classifier currently uses past 16.
            newWords[wInd] = word;

            if (bigrams) {
                if (wInd - windowSize >= 0 && lastWord.getWord() != 0) {
                    BitWordLong bigram = new BitWordLong(newWords[wInd - windowSize], word);

                    ComparablePair<BitWordLong, Byte> key = new ComparablePair<>(bigram, (byte) 0);
                    Integer val = bag.get(key);

                    if (val == null)
                        val = 0;
                    bag.put(key, ++val);
                }
            }

            //add to bag, unless num reduction applies
            if (numerosityReduction && word.equals(lastWord)) {
                ++trivialMatchCount;
                ++wInd;
            }
            else {
                //if a run of equivalent words, those words essentially representing the same
                //elongated pattern. still apply numerosity reduction, however use the central
                //time position to represent its position
                addWordToPyramid(word, wInd - (trivialMatchCount/2), bag);

                lastWord = word;
                trivialMatchCount = 0;
                ++wInd;
            }
        }

        applyPyramidWeights(bag);

        return bag;
    }

    protected void changeNumLevels(int newLevels) {
        //curently, simply remaking bags from words
        //alternatively: un-weight all bags, add(run through SFAwords again)/remove levels, re-weight all

        if (newLevels == this.levels)
            return;

        this.levels = newLevels;

        for (int inst = 0; inst < bags.size(); ++inst) {
            SPBag bag = createSPBagFromWords(wordLength, SFAwords[inst]); //rebuild bag
            bag.setClassVal(bags.get(inst).classVal);
            bags.set(inst, bag); //overwrite old
        }
    }

    protected void applyPyramidWeights(SPBag bag) {
        for (Map.Entry<ComparablePair<BitWordLong, Byte>, Integer> ent : bag.entrySet()) {
            //find level that this quadrant is on
            int quadrant = ent.getKey().var2;
            int qEnd = 0;
            int level = 0;
            while (qEnd < quadrant) {
                int numQuadrants = (int)Math.pow(2, ++level);
                qEnd+=numQuadrants;
            }

            //double val = ent.getValue() * (Math.pow(levelWeighting, levels-level-1)); //weighting ^ (levels - level)
            int val = ent.getValue() * (int)Math.pow(2,level);
            bag.put(ent.getKey(), val);
        }
    }

    protected void addWordToPyramid(BitWordLong word, int wInd, SPBag bag) {
        int qStart = 0; //for this level, whats the start index for quadrants
        //e.g level 0 = 0
        //    level 1 = 1
        //    level 2 = 3
        for (int l = 0; l < levels; ++l) {
            //need to do the cell finding thing in the regular grid
            int numQuadrants = (int)Math.pow(2, l);
            int quadrantSize = seriesLength / numQuadrants;
            int pos = wInd + (windowSize/2); //use the middle of the window as its position
            int quadrant = qStart + (pos/quadrantSize);

            ComparablePair<BitWordLong, Byte> key = new ComparablePair<>(word, (byte)quadrant);
            Integer val = bag.get(key);

            if (val == null)
                val = 0;
            bag.put(key, ++val);

            qStart += numQuadrants;
        }
    }

    protected BitWordLong[] createSFAwords(Instance inst) {
        double[][] dfts = performMFT(toArrayNoClass(inst)); //approximation
        BitWordLong[] words = new BitWordLong[dfts.length];
        for (int window = 0; window < dfts.length; ++window) {
            if (anova) words[window] = createWordAnova(dfts[window]);//discretisation
            else words[window] = createWord(dfts[window]);//discretisation
        }

        return words;
    }

    // Modified version of the trainChiSquared method created by Patrick Schaefer in the WEASEL class
    protected void chiSquared(){
        ObjectIntHashMap<ComparablePair<BitWordLong, Byte>> featureCount = new ObjectIntHashMap<>();
        DoubleIntHashMap classProb = new DoubleIntHashMap();
        DoubleObjectHashMap<ObjectIntHashMap<ComparablePair<BitWordLong, Byte>>> observed = new DoubleObjectHashMap<>();

        // count number of samples with this word
        for (SPBag bag : bags) {
            double label = bag.classVal;

            // samples per class
            classProb.putOrAdd(label, 1, 1);

            int index = observed.indexOf(label);
            ObjectIntHashMap<ComparablePair<BitWordLong, Byte>> obs;
            if (index > -1) {
                obs = observed.indexGet(index);
            } else {
                obs = new ObjectIntHashMap<>();
                observed.put(label, obs);
            }

            for (Map.Entry<ComparablePair<BitWordLong, Byte>, Integer> entry : bag.entrySet()) {
                featureCount.putOrAdd(entry.getKey(), 1,1); //word.value, word.value);

                // count observations per class for this feature
                obs.putOrAdd(entry.getKey(), 1,1); //word.value, word.value);
            }
        }

        // p_value-squared: observed minus expected occurrence
        ObjectDoubleHashMap<ComparablePair<BitWordLong, Byte>> chiSquareSum = new ObjectDoubleHashMap<>(featureCount.size());

        for (DoubleIntCursor prob : classProb) {
            double p = ((double)prob.value) / bags.size();
            ObjectIntHashMap<ComparablePair<BitWordLong, Byte>> obs = observed.get(prob.key);

            for (ObjectIntCursor<ComparablePair<BitWordLong, Byte>> feature : featureCount) {
                double expected = p * feature.value;

                double chi = obs.get(feature.key) - expected;
                double newChi = chi * chi / expected;

                if (newChi > 0) {
                    // build the sum among p_value-values of all classes
                    chiSquareSum.putOrAdd(feature.key, newChi, newChi);
                }
            }
        }

        chiSquare = new ObjectHashSet(featureCount.size());
        ArrayList<ComparablePair<Double, ComparablePair<BitWordLong, Byte>>> values = new ArrayList<>(featureCount.size());

        for (ObjectDoubleCursor<ComparablePair<BitWordLong, Byte>> feature : chiSquareSum) {
            double newChi = feature.value;
            double pvalue = Statistics.chiSquaredProbability(newChi, classProb.keys().size()-1);

            if (pvalue <= chiLimit) {
                chiSquare.add(feature.key);
                values.add(new ComparablePair(pvalue, feature.key));
            }
        }

        // limit number of features avoid excessive features
        int limit = fsLimit;

        System.out.println(featureCount.size() + " " + values.size() + " " + levels + " " + windowSize + " " + wordLength + " " + norm + " " + IGB);

        if (values.size() > limit) {
            // sort by p_value-squared value
            Collections.sort(values);

            chiSquare.clear();

            for (int i = 0; i < limit; i++) {
                chiSquare.add(values.get(i).var2);
            }
        }

        ArrayList<SPBag> newBags = new ArrayList<>(bags.size());

        // remove values
        for (int j = 0; j < bags.size(); j++) {
            SPBag oldBag = bags.get(j);
            SPBag newBag = new SPBag(oldBag.classVal);
            for (Map.Entry<ComparablePair<BitWordLong, Byte>, Integer> entry : oldBag.entrySet()) {
                if (chiSquare.contains(entry.getKey())) {
                    newBag.put(entry.getKey(), entry.getValue());
                }
            }
            newBags.add(newBag);
        }

        bags = newBags;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() != -1 && data.classIndex() != data.numAttributes()-1)
            throw new Exception("BOSS_BuildClassifier: Class attribute not set as last attribute in dataset");

        if (IGB) breakpoints = IGB(data);
        else breakpoints = MCB(data); //breakpoints to be used for making sfa words for train AND test data

        SFAwords = new BitWordLong[data.numInstances()][];
        bags = new ArrayList<>(data.numInstances());
        rand = new Random(seed);
        seriesLength = data.numAttributes()-1;

        if (multiThread){
            if (numThreads == 1) numThreads = Runtime.getRuntime().availableProcessors();
            if (ex == null) ex = Executors.newFixedThreadPool(numThreads);

            ArrayList<Future<SPBag>> futures = new ArrayList<>(data.numInstances());

            for (int inst = 0; inst < data.numInstances(); ++inst)
                futures.add(ex.submit(new TransformThread(inst, data.get(inst))));

            for (Future<SPBag> f: futures)
                bags.add(f.get());
        }
        else {
            for (int inst = 0; inst < data.numInstances(); ++inst) {
                SFAwords[inst] = createSFAwords(data.get(inst));

                SPBag bag = createSPBagFromWords(wordLength, SFAwords[inst]);
                try {
                    bag.setClassVal(data.get(inst).classValue());
                }
                catch(UnassignedClassException e){
                    bag.setClassVal(-1);
                }
                bags.add(bag);
            }
        }

        if (featureSelection) chiSquared();

        if (wekaClassifier) {
            words = new HashMap();

            int idx = 0;
            for (SPBag bag: bags){
                for (ComparablePair<BitWordLong, Byte> word : bag.keySet()){
                    if (!words.containsKey(word)){
                        words.put(word, idx);
                        idx++;
                    }
                }
            }

            ArrayList<Attribute> atts = new ArrayList();
            for (int n = 0; n < words.size(); n++) {
                atts.add(new Attribute("att" + n));
            }
            atts.add(data.classAttribute());

            transformedData = new Instances("Histograms", atts, 0);
            transformedData.setClassIndex(transformedData.numAttributes() - 1);
            header = new Instances(transformedData);

            for (int inst = 0; inst < data.numInstances(); ++inst) {
                SPBag bag = bags.get(inst);
                double[] values = new double[words.size() + 1];

                for (Map.Entry<ComparablePair<BitWordLong, Byte>, Integer> entry : bag.entrySet()) {
                    values[words.get(entry.getKey())] = entry.getValue();
                }
                values[words.size()] = data.get(inst).classValue();

                transformedData.add(new SparseInstance(1, values));
            }

            classifier = new Logistic();
            classifier.buildClassifier(transformedData);
        }

        if (cleanAfterBuild) {
            clean();
        }
    }

    /**
     * Computes BOSSSpatialPyramids distance between two bags d(test, train), is NON-SYMETRIC operation, ie d(a,b) != d(b,a).
     *
     * Quits early if the dist-so-far is greater than bestDist (assumed is in fact the dist still squared), and returns Double.MAX_VALUE
     *
     * @return distance FROM instA TO instB, or Double.MAX_VALUE if it would be greater than bestDist
     */
    public double BOSSSpatialPyramidsDistance(SPBag instA, SPBag instB, double bestDist) {
        double dist = 0.0;

        //find dist only from values in instA
        for (Map.Entry<ComparablePair<BitWordLong, Byte>, Integer> entry : instA.entrySet()) {
            Integer valA = entry.getValue();
            Integer valB = instB.get(entry.getKey());
            if (valB == null)
                valB = 0;
            dist += (valA-valB)*(valA-valB);

            if (dist > bestDist)
                return Double.MAX_VALUE;
        }

        return dist;
    }

    public double histogramIntersection(SPBag instA, SPBag instB) {
        //min vals of keys that exist in only one of the bags will always be 0
        //therefore want to only bother looking at counts of words in both bags
        //therefore will simply loop over words in a, skipping those that dont appear in b
        //no need to loop over b, since only words missed will be those not in a anyway

        double sim = 0.0;

        for (Map.Entry<ComparablePair<BitWordLong, Byte>, Integer> entry : instA.entrySet()) {
            Integer valA = entry.getValue();
            Integer valB = instB.get(entry.getKey());
            if (valB == null)
                continue;

            sim += Math.min(valA,valB);
        }

        return sim;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception{
        BOSSIndividualSP.SPBag testBag = BOSSSpatialPyramidsTransform(instance);;

        if (wekaClassifier){
            double[] values = new double[words.size() + 1];

            for (Map.Entry<ComparablePair<BitWordLong, Byte>, Integer> entry : testBag.entrySet()) {
                Integer idx = words.get(entry.getKey());
                if (idx != null) {
                    values[idx] = entry.getValue();
                }
            }
            values[words.size()] = -1;

            header.add(new SparseInstance(1, values));
            double pred = classifier.classifyInstance(header.firstInstance());
            header.remove(0);
            return pred;
        }
        else if (tuningK){
            //kNN BOSS distance
            PriorityQueue<ComparableKeyPair<Double,Double>> nn = new PriorityQueue<>();

            for (int i = 0; i < bags.size(); ++i) {
                double dist;
                if (histogramIntersection)
                    dist = -histogramIntersection(testBag, bags.get(i));
                else dist = BOSSSpatialPyramidsDistance(testBag, bags.get(i), Double.MAX_VALUE);

                nn.add(new ComparableKeyPair<>(dist, bags.get(i).getClassVal()));
            }

            Iterator<ComparableKeyPair<Double,Double>> it = nn.iterator();
            double[] counts = new double[numClasses];

            for (int i = 0; i < bestK; i++) {
                counts[it.next().var2.intValue()]++;
            }

            return argMax(counts, rand);
        }
        else {
            //1NN BOSS distance
            double bestDist = Double.MAX_VALUE;
            double nn = 0;

            for (int i = 0; i < bags.size(); ++i) {
                double dist;
                if (histogramIntersection)
                    dist = -histogramIntersection(testBag, bags.get(i));
                else dist = BOSSSpatialPyramidsDistance(testBag, bags.get(i), bestDist);

                if (dist < bestDist) {
                    bestDist = dist;
                    nn = bags.get(i).getClassVal();
                }
            }

            return nn;
        }
    }

    /**
     * Used within BOSSEnsemble as part of a leave-one-out crossvalidation, to skip having to rebuild
     * the classifier every time (since the n histograms would be identical each time anyway), therefore this classifies
     * the instance at the index passed while ignoring its own corresponding histogram
     *
     * @param testIndex index of instance to classify
     * @return classification
     */
    public double classifyInstance(int testIndex) throws Exception{
        BOSSIndividualSP.SPBag testBag = bags.get(testIndex);;

        if (tuningK){
            //kNN BOSS distance
            PriorityQueue<ComparableKeyPair<Double,Double>> nn = new PriorityQueue<>();

            for (int i = 0; i < bags.size(); ++i) {
                if (i == testIndex) //skip 'this' one, leave-one-out
                    continue;

                double dist;
                if (histogramIntersection)
                    dist = -histogramIntersection(testBag, bags.get(i));
                else dist = BOSSSpatialPyramidsDistance(testBag, bags.get(i), Double.MAX_VALUE);

                nn.add(new ComparableKeyPair<>(dist, bags.get(i).getClassVal()));
            }

            neighbours.add(nn);
            return nn.peek().var2;
        }
        else {
            //1NN BOSS distance
            double bestDist = Double.MAX_VALUE;
            double nn = 0;

            for (int i = 0; i < bags.size(); ++i) {
                if (i == testIndex) //skip 'this' one, leave-one-out
                    continue;


                double dist;
                if (histogramIntersection)
                    dist = -histogramIntersection(testBag, bags.get(i));
                else dist = BOSSSpatialPyramidsDistance(testBag, bags.get(i), bestDist);

                if (dist < bestDist) {
                    bestDist = dist;
                    nn = bags.get(i).getClassVal();
                }
            }

            return nn;
        }
    }

    public double trainAcc() throws Exception{
        CrossValidationEvaluator cv = new CrossValidationEvaluator();
        cv.setSeed(seed);
        cv.setNumFolds(10);
        double result = cv.crossValidateWithStats(new Logistic(), transformedData).getAcc();
        transformedData = null;
        return result;
    }

    public void tuneK(){
        int mostCorrect = 0;

        if (bags.size() < maxK) maxK = bags.size()-1;

        for (int k = 1; k <= maxK; k+=2){
            int correct = 0;

            for (int i = 0; i < bags.size(); i++){
                Iterator<ComparableKeyPair<Double,Double>> it = neighbours.get(i).iterator();
                int sameClass = 0;

                for (int n = 0; n < k; n++) {
                    if (it.next().var2 == bags.get(i).classVal){
                        sameClass++;

                        if (sameClass > k/2){
                            correct++;
                            break;
                        }
                    }
                }
            }

            if (correct > mostCorrect){
                mostCorrect = correct;
                bestK = k;
            }
        }
    }

    public double FCNN(boolean FCNNcomp, int FCNNsoftlimit) {
        double[][] distances = new double[bags.size()][bags.size()];
        for (int i = 0; i < bags.size(); i++) {
            for (int n = 0; n < bags.size(); n++) {
                if (histogramIntersection) distances[i][n] = -histogramIntersection(bags.get(i), bags.get(n));
                else distances[i][n] = BOSSSpatialPyramidsDistance(bags.get(i), bags.get(n), Double.MAX_VALUE);
            }
        }

        HashSet<Integer> S = new HashSet<>();
        HashSet<Integer> S2 = new HashSet<>();
        ArrayList<Integer> T = new ArrayList<>(bags.size());
        for (int i = 0; i < bags.size(); i++) {
            T.add(i);
        }

        while (S.size() < FCNNsoftlimit){
            //centroids
            for (int c = 0; c < numClasses; c++) {
                double minDist = Double.MAX_VALUE;
                int bestMedoid = -1;

                for (int i = 0; i < T.size(); i++) {
                    if (bags.get(T.get(i)).classVal != c)
                        continue;

                    double medoidDist = 0;

                    for (int n = 0; n < T.size(); n++) {
                        if (bags.get(T.get(n)).classVal != c)
                            continue;

                        medoidDist += distances[T.get(i)][T.get(n)];
                    }

                    if (medoidDist < minDist) {
                        minDist = medoidDist;
                        bestMedoid = T.get(i);
                    }
                }

                if (bestMedoid != -1) {
                    S2.add(bestMedoid);
                }
            }

            int[] nearest = new int[bags.size()];

            while (!S2.isEmpty()) {
                S.addAll(S2);
                T.removeAll(S2);
                int[] rep = new int[bags.size()];
                for (int i = 0; i < bags.size(); i++) {
                    rep[i] = -1;
                }

                for (Integer q : T) {
                    for (Integer p : S2) {
                        if (distances[nearest[q]][q] < distances[p][q]) {
                            nearest[q] = p;
                        }
                    }

                    if (bags.get(q).classVal != bags.get(nearest[q]).classVal &&
                            (rep[nearest[q]] == -1 || distances[nearest[q]][q] < distances[nearest[q]][rep[nearest[q]]])) {
                        rep[nearest[q]] = q;
                    }
                }

                S2 = new HashSet<>();
                for (Integer p : S) {
                    if (rep[p] != -1) {
                        S2.add(rep[p]);
                    }
                }
            }
        }

        double correct = 0;
        for (int i = 0; i < bags.size(); ++i) {
            double bestDist = Double.MAX_VALUE;
            double nn = 0;

            for (Integer p: S) {
                if (p == i) //skip 'this' one, leave-one-out
                    continue;

                if (distances[i][p] < bestDist) {
                    bestDist = distances[i][p];
                    nn = bags.get(p).getClassVal();
                }
            }

            if (bags.get(i).classVal == nn)
                correct++;
        }
        double acc = correct/bags.size();

        if (FCNNcomp) {
            double correct2 = 0;
            for (int i = 0; i < bags.size(); ++i) {
                double bestDist = Double.MAX_VALUE;
                double nn = 0;

                for (int p = 0; p < bags.size(); ++p) {
                    if (p == i) //skip 'this' one, leave-one-out
                        continue;

                    if (distances[i][p] < bestDist) {
                        bestDist = distances[i][p];
                        nn = bags.get(p).getClassVal();
                    }
                }

                if (bags.get(i).classVal == nn)
                    correct2++;
            }
            double acc2 = correct2 / bags.size();

            if (acc2 <= acc) {
                ArrayList<SPBag> newBags = new ArrayList<>(S.size());
                for (Integer p : S) {
                    newBags.add(bags.get(p));
                }
                bags = newBags;

                return acc;
            }

            return acc2;
        }
        else {
            ArrayList<SPBag> newBags = new ArrayList<>(S.size());
            for (Integer p : S) {
                newBags.add(bags.get(p));
            }
            bags = newBags;

            return acc;
        }
    }

    public class TestNearestNeighbourThread implements Callable<Double>{
        Instance inst;

        public TestNearestNeighbourThread(Instance inst){
            this.inst = inst;
        }

        @Override
        public Double call() {
            BOSSIndividualSP.SPBag testBag = BOSSSpatialPyramidsTransform(inst);

            //1NN BOSS distance
            double bestDist = Double.MAX_VALUE;
            double nn = 0;

            for (int i = 0; i < bags.size(); ++i) {
                double dist = BOSSSpatialPyramidsDistance(testBag, bags.get(i), bestDist);

                if (dist < bestDist) {
                    bestDist = dist;
                    nn = bags.get(i).getClassVal();
                }
            }

            return nn;
        }
    }

    public class TrainNearestNeighbourThread implements Callable<Double>{
        int testIndex;

        public TrainNearestNeighbourThread(int testIndex){
            this.testIndex = testIndex;
        }

        @Override
        public Double call() {
            BOSSIndividualSP.SPBag testBag = bags.get(testIndex);

            //1NN BOSS distance
            double bestDist = Double.MAX_VALUE;
            double nn = 0;

            for (int i = 0; i < bags.size(); ++i) {
                if (i == testIndex) //skip 'this' one, leave-one-out
                    continue;

                double dist = BOSSSpatialPyramidsDistance(testBag, bags.get(i), bestDist);

                if (dist < bestDist) {
                    bestDist = dist;
                    nn = bags.get(i).getClassVal();
                }
            }

            return nn;
        }
    }

    private class TransformThread implements Callable<SPBag>{
        int i;
        Instance inst;

        public TransformThread(int i, Instance inst){
            this.i = i;
            this.inst = inst;
        }

        @Override
        public SPBag call() {
            SFAwords[i] = createSFAwords(inst);

            SPBag bag = createSPBagFromWords(wordLength, SFAwords[i]);
            try {
                bag.setClassVal(inst.classValue());
            }
            catch(UnassignedClassException e){
                bag.setClassVal(-1);
            }

            return bag;
        }
    }
}
