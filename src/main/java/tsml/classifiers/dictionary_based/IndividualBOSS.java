package tsml.classifiers.dictionary_based;

import tsml.classifiers.MultiThreadable;
import tsml.classifiers.dictionary_based.bitword.BitWordInt;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.UnassignedClassException;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

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
public class IndividualBOSS extends AbstractClassifier implements Serializable, Comparable<IndividualBOSS>, MultiThreadable {

    //all sfa words found in original buildClassifier(), no numerosity reduction/shortening applied
    protected BitWordInt[/*instance*/][/*windowindex*/] SFAwords;

    //histograms of words of the current wordlength with numerosity reduction applied (if selected)
    protected ArrayList<Bag> bags;

    //breakpoints to be found by MCB
    protected double[/*letterindex*/][/*breakpointsforletter*/] breakpoints;

    protected double inverseSqrtWindowSize;
    protected int windowSize;
    protected int wordLength;
    protected int alphabetSize;
    protected boolean norm;
    protected boolean numerosityReduction = true;
    protected boolean cleanAfterBuild = false;

    protected double accuracy = -1;
    protected double weight = 1;
    protected ArrayList<Integer> subsampleIndices;

    protected boolean multiThread = false;
    protected int numThreads = 1;
    protected ExecutorService ex;

    protected int seed = 0;
    protected Random rand;

    protected static final long serialVersionUID = 22551L;

    public IndividualBOSS(int wordLength, int alphabetSize, int windowSize, boolean normalise, boolean multiThread, int numThreads, ExecutorService ex) {
        this.wordLength = wordLength;
        this.alphabetSize = alphabetSize;
        this.windowSize = windowSize;
        this.inverseSqrtWindowSize = 1.0 / Math.sqrt(windowSize);
        this.norm = normalise;
        this.multiThread = multiThread;
        this.numThreads = numThreads;
        this.ex = ex;
    }

    public IndividualBOSS(int wordLength, int alphabetSize, int windowSize, boolean normalise) {
        this.wordLength = wordLength;
        this.alphabetSize = alphabetSize;
        this.windowSize = windowSize;
        this.inverseSqrtWindowSize = 1.0 / Math.sqrt(windowSize);
        this.norm = normalise;
    }

    /**
     * Used when shortening histograms, copies 'meta' data over, but with shorter
     * word length, actual shortening happens separately
     */
    public IndividualBOSS(IndividualBOSS boss, int wordLength) {
        this.wordLength = wordLength;

        this.windowSize = boss.windowSize;
        this.inverseSqrtWindowSize = boss.inverseSqrtWindowSize;
        this.alphabetSize = boss.alphabetSize;
        this.norm = boss.norm;
        this.numerosityReduction = boss.numerosityReduction;

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
    public int compareTo(IndividualBOSS o) {
        return Double.compare(this.accuracy, o.accuracy);
    }

    @Override
    public void enableMultiThreading(int numThreads) {
        this.numThreads = numThreads;
    }

    public static class Bag extends HashMap<BitWordInt, Integer> {
        double classVal;
        protected static final long serialVersionUID = 22552L;

        public Bag() {
            super();
        }

        public Bag(int classValue) {
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

    public ArrayList<Bag> getBags() { return bags; }

    /**
     * @return { numIntervals(word length), alphabetSize, slidingWindowSize, normalise? }
     */
    public int[] getParameters() {
        return new int[] { wordLength, alphabetSize, windowSize };
    }

    public void setSeed(int i){ seed = i; }

    public void clean() {
        SFAwords = null;
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

        int sample = 0;
        for (Instance inst : data)
            dfts[sample++] = performDFT(disjointWindows(toArrayNoClass(inst))); //approximation

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

    /**
     * Builds a brand new boss bag from the passed fourier transformed data, rather than from
     * looking up existing transforms from earlier builds (i.e. SFAWords).
     *
     * to be used e.g to transform new test instances
     */
    protected Bag createBagSingle(double[][] dfts) {
        Bag bag = new Bag();
        BitWordInt lastWord = new BitWordInt();

        for (double[] d : dfts) {
            BitWordInt word = createWord(d);
            //add to bag, unless num reduction applies
            if (numerosityReduction && word.equals(lastWord))
                continue;

            Integer val = bag.get(word);
            if (val == null)
                val = 0;
            bag.put(word, ++val);

            lastWord = word;
        }

        return bag;
    }

    protected BitWordInt createWord(double[] dft) {
        BitWordInt word = new BitWordInt();
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
     * @return BOSSTransform-ed bag, built using current parameters
     */
    public Bag BOSSTransform(Instance inst) {
        double[][] mfts = performMFT(toArrayNoClass(inst)); //approximation
        Bag bag = createBagSingle(mfts); //discretisation/bagging
        bag.setClassVal(inst.classValue());

        return bag;
    }

    /**
     * Shortens all bags in this BOSS instance (histograms) to the newWordLength, if wordlengths
     * are same, instance is UNCHANGED
     *
     * @param newWordLength wordLength to shorten it to
     * @return new boss classifier with newWordLength, or passed in classifier if wordlengths are same
     */
    public IndividualBOSS buildShortenedBags(int newWordLength) throws Exception {
        if (newWordLength == wordLength) //case of first iteration of word length search in ensemble
            return this;
        if (newWordLength > wordLength)
            throw new Exception("Cannot incrementally INCREASE word length, current:"+wordLength+", requested:"+newWordLength);
        if (newWordLength < 2)
            throw new Exception("Invalid wordlength requested, current:"+wordLength+", requested:"+newWordLength);

        IndividualBOSS newBoss = new IndividualBOSS(this, newWordLength);

        //build hists with new word length from SFA words, and copy over the class values of original insts
        for (int i = 0; i < bags.size(); ++i) {
            Bag newBag = createBagFromWords(newWordLength, SFAwords[i]);
            newBag.setClassVal(bags.get(i).getClassVal());
            newBoss.bags.add(newBag);
        }

        return newBoss;
    }

    /**
     * Builds a bag from the set of words for a pre-transformed series of a given wordlength.
     */
    protected Bag createBagFromWords(int thisWordLength, BitWordInt[] words) {
        Bag bag = new Bag();
        BitWordInt lastWord = new BitWordInt();

        for (BitWordInt w : words) {
            BitWordInt word = new BitWordInt(w);
            if (wordLength != thisWordLength)
                word.shorten(BitWordInt.MAX_LENGTH-thisWordLength);

            //add to bag, unless num reduction applies
            if (numerosityReduction && word.equals(lastWord))
                continue;

            Integer val = bag.get(word);
            if (val == null)
                val = 0;
            bag.put(word, ++val);

            lastWord = word;
        }

        return bag;
    }

    protected BitWordInt[] createSFAwords(Instance inst) {
        double[][] dfts = performMFT(toArrayNoClass(inst)); //approximation
        BitWordInt[] words = new BitWordInt[dfts.length];
        for (int window = 0; window < dfts.length; ++window)
            words[window] = createWord(dfts[window]);//discretisation

        return words;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() != -1 && data.classIndex() != data.numAttributes()-1)
            throw new Exception("BOSS_BuildClassifier: Class attribute not set as last attribute in dataset");

        breakpoints = MCB(data); //breakpoints to be used for making sfa words for train AND test data
        SFAwords = new BitWordInt[data.numInstances()][];
        bags = new ArrayList<>(data.numInstances());
        rand = new Random(seed);

        if (multiThread){
            if (numThreads == 1) numThreads = Runtime.getRuntime().availableProcessors();
            if (ex == null) ex = Executors.newFixedThreadPool(numThreads);

            ArrayList<Future<Bag>> futures = new ArrayList<>(data.numInstances());

            for (int inst = 0; inst < data.numInstances(); ++inst)
                futures.add(ex.submit(new TransformThread(inst, data.get(inst))));

            for (Future<Bag> f: futures)
                bags.add(f.get());
        }
        else {
            for (int inst = 0; inst < data.numInstances(); ++inst) {
                SFAwords[inst] = createSFAwords(data.get(inst));

                Bag bag = createBagFromWords(wordLength, SFAwords[inst]);
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
     * Computes BOSS distance between two bags d(test, train), is NON-SYMETRIC operation, ie d(a,b) != d(b,a).
     *
     * Quits early if the dist-so-far is greater than bestDist (assumed dist is still the squared distance), and returns Double.MAX_VALUE
     *
     * @return distance FROM instA TO instB, or Double.MAX_VALUE if it would be greater than bestDist
     */
    public double BOSSdistance(Bag instA, Bag instB, double bestDist) {
        double dist = 0.0;

        //find dist only from values in instA
        for (Map.Entry<BitWordInt, Integer> entry : instA.entrySet()) {
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

    @Override
    public double classifyInstance(Instance instance) throws Exception{
        IndividualBOSS.Bag testBag = BOSSTransform(instance);

        //1NN BOSS distance
        double bestDist = Double.MAX_VALUE;
        double nn = -1;

        for (int i = 0; i < bags.size(); ++i) {
            double dist = BOSSdistance(testBag, bags.get(i), bestDist);

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
        IndividualBOSS.Bag testBag = bags.get(testIndex);

        //1NN BOSS distance
        double bestDist = Double.MAX_VALUE;
        double nn = -1;

        for (int i = 0; i < bags.size(); ++i) {
            if (i == testIndex) //skip 'this' one, leave-one-out
                continue;

            double dist = BOSSdistance(testBag, bags.get(i), bestDist);

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
            IndividualBOSS.Bag testBag = BOSSTransform(inst);

            //1NN BOSS distance
            double bestDist = Double.MAX_VALUE;
            double nn = -1;

            for (int i = 0; i < bags.size(); ++i) {
                double dist = BOSSdistance(testBag, bags.get(i), bestDist);

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
            IndividualBOSS.Bag testBag = bags.get(testIndex);

            //1NN BOSS distance
            double bestDist = Double.MAX_VALUE;
            double nn = -1;

            for (int i = 0; i < bags.size(); ++i) {
                if (i == testIndex) //skip 'this' one, leave-one-out
                    continue;

                double dist = BOSSdistance(testBag, bags.get(i), bestDist);

                if (dist < bestDist) {
                    bestDist = dist;
                    nn = bags.get(i).getClassVal();
                }
            }

            return nn;
        }
    }

    private class TransformThread implements Callable<Bag>{
        int i;
        Instance inst;

        public TransformThread(int i, Instance inst){
            this.i = i;
            this.inst = inst;
        }

        @Override
        public Bag call() {
            SFAwords[i] = createSFAwords(inst);

            Bag bag = createBagFromWords(wordLength, SFAwords[i]);
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
