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
package tsml.classifiers.dictionary_based;

import com.carrotsearch.hppc.*;
import com.carrotsearch.hppc.cursors.IntCursor;
import tsml.classifiers.MultiThreadable;
import tsml.classifiers.dictionary_based.bitword.BitWord;
import tsml.classifiers.dictionary_based.bitword.BitWordInt;
import tsml.classifiers.dictionary_based.bitword.BitWordLong;
import utilities.generic_storage.SerialisableComparablePair;
import weka.classifiers.AbstractClassifier;
import weka.core.*;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Improved BOSS classifier to be used with known parameters, for ensemble use TDE.
 *
 * Current implementation of BitWord as of 18/03/2020 only supports alphabetsize of 4, which is the expected value
 * as defined in the original BOSS paper
 *
 * Params: wordLength, alphabetSize, windowLength, normalise, levels, IGB
 */
public class IndividualTDE extends AbstractClassifier implements Serializable, Comparable<IndividualTDE>,
        MultiThreadable {

    //all sfa words found in original buildClassifier(), no numerosity reduction/shortening applied
    private BitWord[/*instance*/][/*windowindex*/] SFAwords;

    //histograms of words of the current wordlength with numerosity reduction applied (if selected)
    private ArrayList<SPBag> bags;

    //breakpoints to be found by MCB or IGB
    private double[/*letterindex*/][/*breakpointsforletter*/] breakpoints;

    private int windowSize;
    private int wordLength;
    private int alphabetSize;
    private boolean norm;
    private int levels;
    private boolean IGB;

    private boolean histogramIntersection = true;
    private boolean useBigrams = true;
    private double levelWeighting = 0.5;
    private boolean numerosityReduction = true;
    private double inverseSqrtWindowSize;
    private boolean cleanAfterBuild = false;
    private int seriesLength;

    private int ensembleID = -1;
    private double accuracy = -1;
    private double weight = 1;
    private ArrayList<Integer> subsampleIndices;
    private ArrayList<Integer> trainPreds;

    private boolean multiThread = false;
    private int numThreads = 1;
    private ExecutorService ex;

    private int seed = 0;
    private Random rand;

    private static final long serialVersionUID = 1L;

    public IndividualTDE(int wordLength, int alphabetSize, int windowSize, boolean normalise, int levels, boolean IGB,
                         boolean multiThread, int numThreads, ExecutorService ex) {
        this.wordLength = wordLength;
        this.alphabetSize = alphabetSize;
        this.windowSize = windowSize;
        this.inverseSqrtWindowSize = 1.0 / Math.sqrt(windowSize);
        this.norm = normalise;
        this.levels = levels;
        this.IGB = IGB;
        this.multiThread = multiThread;
        this.numThreads = numThreads;
        this.ex = ex;
    }

    public IndividualTDE(int wordLength, int alphabetSize, int windowSize, boolean normalise, int levels, boolean IGB) {
        this.wordLength = wordLength;
        this.alphabetSize = alphabetSize;
        this.windowSize = windowSize;
        this.inverseSqrtWindowSize = 1.0 / Math.sqrt(windowSize);
        this.norm = normalise;
        this.levels = levels;
        this.IGB = IGB;
    }

    /**
     * Used when shortening histograms, copies 'meta' data over, but with shorter
     * word length, actual shortening happens separately
     */
    public IndividualTDE(IndividualTDE boss, int wordLength) {
        this.wordLength = wordLength;

        this.windowSize = boss.windowSize;
        this.inverseSqrtWindowSize = boss.inverseSqrtWindowSize;
        this.alphabetSize = boss.alphabetSize;
        this.norm = boss.norm;
        this.levels = boss.levels;

        this.histogramIntersection = boss.histogramIntersection;
        this.levelWeighting = boss.levelWeighting;
        this.numerosityReduction = boss.numerosityReduction;
        this.inverseSqrtWindowSize = boss.inverseSqrtWindowSize;
        this.cleanAfterBuild = boss.cleanAfterBuild;
        this.seriesLength = boss.seriesLength;

        this.multiThread = boss.multiThread;
        this.numThreads = boss.numThreads;
        this.ex = boss.ex;

        this.seed = boss.seed;
        this.rand = boss.rand;

        this.SFAwords = boss.SFAwords;
        this.bags = new ArrayList<>(boss.bags.size());
        this.breakpoints = boss.breakpoints;
    }

    @Override
    public int compareTo(IndividualTDE o) {
        return Double.compare(this.accuracy, o.accuracy);
    }

    @Override
    public void enableMultiThreading(int numThreads) {
        this.numThreads = numThreads;
    }

    //map of <word, level> => count
    public static class SPBag extends HashMap<SerialisableComparablePair<BitWord, Byte>, Integer> {
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
    public boolean getNorm() { return norm; }
    public int getLevels() { return levels; }
    public boolean getIGB() { return IGB; }
    public ArrayList<SPBag> getBags() { return bags; }
    public int getEnsembleID() { return ensembleID; }
    public double getAccuracy() { return accuracy; }
    public double getWeight() { return weight; }
    public ArrayList<Integer> getSubsampleIndices() { return subsampleIndices; }
    public ArrayList<Integer> getTrainPreds() { return trainPreds; }

    public void setSeed(int i){ seed = i; }
    public void setCleanAfterBuild(boolean b){ cleanAfterBuild = b; }
    public void setEnsembleID(int i) { ensembleID = i; }
    public void setAccuracy(double d) { accuracy = d; }
    public void setWeight(double d) { weight = d; }
    public void setSubsampleIndices(ArrayList<Integer> arr) { subsampleIndices = arr; }
    public void setTrainPreds(ArrayList<Integer> arr) { trainPreds = arr; }
    public void setHistogramIntersection(boolean b) { histogramIntersection = b; }
    public void setUseBigrams(boolean b) { useBigrams = b; }

    public void clean() {
        SFAwords = null;
    }

    private double[][] performDFT(double[][] windows) {
        double[][] dfts = new double[windows.length][wordLength];
        for (int i = 0; i < windows.length; ++i) {
            dfts[i] = DFT(windows[i]);
        }
        return dfts;
    }

    private double stdDev(double[] series) {
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

    private double[] DFT(double[] series) {
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
        int outputLength = wordLength/2;
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
        // ignore DC value?
        int startOffset = norm ? 2 : 0;
        int l = wordLength;
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

    private double[][] disjointWindows(double [] data) {
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

    private double[][] MCB(Instances data) {
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

        return breakpoints;
    }

    //IGB code by Patrick Schaefer from the WEASEL class
    private double[][] IGB(Instances data) {
        ArrayList<SerialisableComparablePair<Double,Double>>[] orderline = new ArrayList[wordLength];
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

                    orderline[n].add(new SerialisableComparablePair<>(value, inst.classValue()));
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

        return breakpoints;
    }

    private void findBestSplit(List<SerialisableComparablePair<Double,Double>> element, int start, int end, int remainingSymbols,
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

    private double entropy(ObjectIntHashMap<Double> frequency, double total) {
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

    private double calculateInformationGain(ObjectIntHashMap<Double> cIn, ObjectIntHashMap<Double> cOut,
                                              double class_entropy, double total_c_in, double total) {
        double total_c_out = (total - total_c_in);
        return class_entropy
                - total_c_in / total * entropy(cIn, total_c_in)
                - total_c_out / total * entropy(cOut, total_c_out);
    }

    private int moveElement(List<SerialisableComparablePair<Double,Double>> element, ObjectIntHashMap<Double> cIn,
                              ObjectIntHashMap<Double> cOut, int pos) {
        cIn.putOrAdd(element.get(pos).var2, 1, 1);
        cOut.putOrAdd(element.get(pos).var2, -1, -1);
        return 1;
    }

    /**
     * Builds a brand new boss bag from the passed fourier transformed data, rather than from
     * looking up existing transforms from earlier builds (i.e. SFAWords).
     *
     * to be used e.g to transform new test instances
     */
    private SPBag createSPBagSingle(double[][] dfts) {
        SPBag bag = new SPBag();
        BitWord lastWord = new BitWordInt();
        BitWord[] words = new BitWord[dfts.length];

        int wInd = 0;
        int trivialMatchCount = 0;

        for (double[] d : dfts) {
            BitWord word = createWord(d);
            words[wInd] = word;

            if (useBigrams) {
                if (wInd - windowSize >= 0) {
                    BitWord bigram = new BitWordLong(words[wInd - windowSize], word);

                    SerialisableComparablePair<BitWord, Byte> key = new SerialisableComparablePair<>(bigram, (byte) -1);
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

    private BitWord createWord(double[] dft) {
        BitWord word = new BitWordInt();
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
    private static double[] toArrayNoClass(Instance inst) {
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
    public IndividualTDE buildShortenedSPBags(int newWordLength) throws Exception {
        if (newWordLength == wordLength) //case of first iteration of word length search in ensemble
            return this;
        if (newWordLength > wordLength)
            throw new Exception("Cannot incrementally INCREASE word length, current:"+wordLength+", requested:"
                    +newWordLength);
        if (newWordLength < 2)
            throw new Exception("Invalid wordlength requested, current:"+wordLength+", requested:"+newWordLength);

        IndividualTDE newBoss = new IndividualTDE(this, newWordLength);

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
    private SPBag createSPBagFromWords(int thisWordLength, BitWord[] words) {
        SPBag bag = new SPBag();
        BitWord lastWord = new BitWordInt();
        BitWord[] newWords = new BitWord[words.length];

        int wInd = 0;
        int trivialMatchCount = 0; //keeps track of how many words have been the same so far

        for (BitWord w : words) {
            BitWord word = new BitWordInt(w);
            if (wordLength != thisWordLength)
                word.shorten(16-thisWordLength); //max word length, no classifier currently uses past 16.
            newWords[wInd] = word;

            if (useBigrams) {
                if (wInd - windowSize >= 0) {
                    BitWord bigram = new BitWordLong(newWords[wInd - windowSize], word);

                    SerialisableComparablePair<BitWord, Byte> key = new SerialisableComparablePair<>(bigram, (byte) -1);
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

    private void changeNumLevels(int newLevels) {
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

    private void applyPyramidWeights(SPBag bag) {
        for (Map.Entry<SerialisableComparablePair<BitWord, Byte>, Integer> ent : bag.entrySet()) {
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

    private void addWordToPyramid(BitWord word, int wInd, SPBag bag) {
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

            SerialisableComparablePair<BitWord, Byte> key = new SerialisableComparablePair<>(word, (byte)quadrant);
            Integer val = bag.get(key);

            if (val == null)
                val = 0;
            bag.put(key, ++val);

            qStart += numQuadrants;
        }
    }

    private BitWord[] createSFAwords(Instance inst) {
        double[][] dfts = performMFT(toArrayNoClass(inst)); //approximation
        BitWord[] words = new BitWord[dfts.length];
        for (int window = 0; window < dfts.length; ++window) {
            words[window] = createWord(dfts[window]);//discretisation
        }

        return words;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() != -1 && data.classIndex() != data.numAttributes()-1)
            throw new Exception("TDE_BuildClassifier: Class attribute not set as last attribute in dataset");

        if (IGB) breakpoints = IGB(data);
        else breakpoints = MCB(data); //breakpoints to be used for making sfa words for train AND test data

        SFAwords = new BitWord[data.numInstances()][];
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

        if (cleanAfterBuild) {
            clean();
        }
    }

    /**
     * Computes BOSSSpatialPyramids distance between two bags d(test, train), is NON-SYMETRIC operation,
     * ie d(a,b) != d(b,a).
     *
     * Quits early if the dist-so-far is greater than bestDist (assumed is in fact the dist still squared),
     * and returns Double.MAX_VALUE
     *
     * @return distance FROM instA TO instB, or Double.MAX_VALUE if it would be greater than bestDist
     */
    public double BOSSSpatialPyramidsDistance(SPBag instA, SPBag instB, double bestDist) {
        double dist = 0.0;

        //find dist only from values in instA
        for (Map.Entry<SerialisableComparablePair<BitWord, Byte>, Integer> entry : instA.entrySet()) {
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

        for (Map.Entry<SerialisableComparablePair<BitWord, Byte>, Integer> entry : instA.entrySet()) {
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
        IndividualTDE.SPBag testBag = BOSSSpatialPyramidsTransform(instance);;

        //1NN distance
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

    /**
     * Used within BOSSEnsemble as part of a leave-one-out crossvalidation, to skip having to rebuild
     * the classifier every time (since the n histograms would be identical each time anyway), therefore this classifies
     * the instance at the index passed while ignoring its own corresponding histogram
     *
     * @param testIndex index of instance to classify
     * @return classification
     */
    public double classifyInstance(int testIndex) throws Exception{
        IndividualTDE.SPBag testBag = bags.get(testIndex);

        //1NN distance
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

    public class TestNearestNeighbourThread implements Callable<Double>{
        Instance inst;

        public TestNearestNeighbourThread(Instance inst){
            this.inst = inst;
        }

        @Override
        public Double call() {
            IndividualTDE.SPBag testBag = BOSSSpatialPyramidsTransform(inst);

            //1NN distance
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
            IndividualTDE.SPBag testBag = bags.get(testIndex);

            //1NN distance
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
