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
package tsml.classifiers.multivariate;

import com.carrotsearch.hppc.*;
import com.carrotsearch.hppc.cursors.*;
import de.bwaldvogel.liblinear.*;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;
import experiments.data.DatasetLoading;
import tsml.classifiers.EnhancedAbstractClassifier;
import utilities.ClassifierTools;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import static utilities.multivariate_tools.MultivariateInstanceTools.*;

/**
 * The WEASEL+MUSE classifier as published in
 *
 * Schäfer, P., Leser, U.: Multivariate Time Series Classification
 * with WEASEL+MUSE. arXiv 2017
 * http://arxiv.org/abs/1711.11343
 *
 * Code adapted from the tsml WEASEL code and WEASEL+MUSE implementation
 * in the SFA package by Patrick Schäfer
 * https://github.com/patrickzib/SFA
 *
 * Author: Matthew Middlehurst 29/07/2020
 */
public class WEASEL_MUSE extends EnhancedAbstractClassifier {

    private static int maxF = 6;
    private static int minF = 2;
    private static int maxS = 4;
    private static boolean[] NORMALIZATION = new boolean[]{true, false};

    private enum HistogramType {
        EQUI_FREQUENCY, EQUI_DEPTH
    }
    private static HistogramType[] histTypes
            = new HistogramType[]{HistogramType.EQUI_DEPTH, HistogramType.EQUI_FREQUENCY};

    private static double chi = 2;
    private static double bias = 1;
    private static SolverType solverType = SolverType.L2R_LR;
    private static int iterations = 5000;
    private static double p = 0.1;
    private static double c = 1;

    private Instances header;
    private boolean derivatives = true;

    // ten-fold cross validation
    private int folds = 10;

    private static int MIN_WINDOW_LENGTH = 2;
    private static int MAX_WINDOW_LENGTH = 450;

    private MUSEModel classifier;

    public WEASEL_MUSE() {
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
    }

    @Override
    public void buildClassifier(final Instances samples) throws Exception {
        long t1=System.nanoTime();

        if (samples.classIndex() != samples.numAttributes()-1)
            throw new Exception("WEASEL_MUSE_BuildClassifier: Class attribute not set as last attribute in dataset");

        Instances newSamples;
        //get derivatives for the instances if enabled
        if (derivatives){
            int dimensionality = numDimensions(samples);
            Instances[] split = splitMultivariateInstances(samples);
            Instances[] channels = new Instances[dimensionality * 2];

            for (int i = 0; i < dimensionality; i++) {
                Instances derivative = new Instances(split[i], 0);
                for (int n = 0; n < samples.numInstances(); n++) {
                    Instance inst = split[i].get(n);
                    double[] d = new double[inst.numAttributes()];
                    for (int a = 1; a < inst.numAttributes()-1; a++) {
                        d[a - 1] = Math.abs(inst.value(a) - inst.value(a - 1));
                    }
                    d[inst.numAttributes()-1] = inst.classValue();
                    derivative.add(new DenseInstance(1, d));
                }
                channels[i] = split[i];
                channels[dimensionality + i] = derivative;
            }

            newSamples = mergeToMultivariateInstances(channels);
            header = new Instances(newSamples, 0);
        }
        else{
            newSamples = samples;
        }

        int dimensionality = numDimensions(newSamples);

        try {
            int maxCorrect = -1;
            int bestF = -1;
            boolean bestNorm = false;
            HistogramType bestHistType = null;

            optimize:
            for (final HistogramType histType : histTypes) {
                for (final boolean mean : NORMALIZATION) {
                    int[] windowLengths = getWindowLengths(newSamples, mean);

                    for (int f = minF; f <= maxF; f += 2) {
                        final MUSE model = new MUSE(f, maxS, histType, windowLengths, mean);
                        MUSE.BagOfBigrams[] bag = null;

                        for (int w = 0; w < model.windowLengths.length; w++) {
                            int[][] words = model.createWords(newSamples, w);
                            MUSE.BagOfBigrams[] bobForOneWindow = fitOneWindow(
                                    newSamples,
                                    windowLengths, mean, histType,
                                    model,
                                    words, f, dimensionality, w);
                            bag = mergeBobs(bag, bobForOneWindow);
                        }

                        // train liblinear
                        final Problem problem = initLibLinearProblem(bag, model.dict, bias);
                        int correct = trainLibLinear(problem, solverType, c, iterations, p, folds);

                        if (correct > maxCorrect || correct == maxCorrect && f < bestF) {
                            maxCorrect = correct;
                            bestF = f;
                            bestNorm = mean;
                            bestHistType = histType;

                            if (debug) {
                                System.out.println("New best model" + maxCorrect + " " + bestF + " " + bestNorm + " "
                                        + bestHistType);
                            }
                        }
                        if (correct == newSamples.numInstances()) {
                            break optimize;
                        }
                    }
                }
            }

            // obtain the final matrix
            int[] windowLengths = getWindowLengths(newSamples, bestNorm);

            // obtain the final matrix
            MUSE model = new MUSE(bestF, maxS, bestHistType, windowLengths, bestNorm);
            MUSE.BagOfBigrams[] bob = null;

            for (int w = 0; w < model.windowLengths.length; w++) {
                int[][] words = model.createWords(newSamples, w);

                MUSE.BagOfBigrams[] bobForOneWindow = fitOneWindow(
                        newSamples,
                        windowLengths, bestNorm, bestHistType,
                        model,
                        words,
                        bestF, dimensionality, w);
                bob = mergeBobs(bob, bobForOneWindow);
            }

            // train liblinear
            Problem problem = initLibLinearProblem(bob, model.dict, bias);
            Parameter par = new Parameter(solverType, c, iterations, p);
            //par.setThreadCount(Math.min(Runtime.getRuntime().availableProcessors(),10));
            de.bwaldvogel.liblinear.Model linearModel = Linear.train(problem, par);

            this.classifier = new MUSEModel(
                    bestNorm,
                    bestF,
                    bestHistType,
                    model,
                    linearModel);

        } catch (Exception e) {
            e.printStackTrace();
        }

        long t2=System.nanoTime();
        trainResults.setClassifierName(getClassifierName());
        trainResults.setParas(classifierName);
        trainResults.setBuildTime(t2-t1);
        trainResults.setParas(getParameters());
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        FeatureNode[] features = predictionTransform(instance);
        return Linear.predict(classifier.linearModel, features);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        FeatureNode[] features = predictionTransform(instance);

        double[] probabilities = new double[classifier.linearModel.getNrClass()];
        Linear.predictProbability(classifier.linearModel, features, probabilities);

        double[] classHist = new double[instance.numClasses()];
        for (int i = 0; i < classifier.linearModel.getLabels().length; i++) {
            classHist[classifier.linearModel.getLabels()[i]] = probabilities[i];
        }
        return classHist;
    }

    private FeatureNode[] predictionTransform(Instance instance){
        Instance newInstance;
        //get derivatives for the instance if enabled
        if (derivatives){
            int dimensionality = numDimensions(instance);
            Instance[] split = splitMultivariateInstance(instance);
            double[][] channels = new double[dimensionality * 2][split[0].numAttributes()];

            for (int i = 0; i < dimensionality; i++) {
                for (int a = 1; a < split[i].numAttributes(); a++) {
                    channels[dimensionality + i][a - 1] = Math.abs(split[i].value(a) - split[i].value(a - 1));
                }
                channels[i] = split[i].toDoubleArray();
            }

            newInstance = new DenseInstance(2);
            Instances relational = createRelationFrom(header.attribute(0).relation(), channels);

            newInstance.setDataset(header);
            int index = newInstance.attribute(0).addRelation(relational);
            newInstance.setValue(0, index);
            newInstance.setValue(1, instance.classValue());;
        }
        else{
            newInstance = instance;
        }

        int dimensionality = numDimensions(newInstance);

        MUSE.BagOfBigrams[] bagTest = null;
        for (int w = 0; w < classifier.muse.windowLengths.length; w++) {
            int[][] wordsTest = classifier.muse.createWords(newInstance, w);
            MUSE.BagOfBigrams[] bopForWindow = new MUSE.BagOfBigrams[]{classifier.muse.createBagOfPatterns(wordsTest,
                    newInstance, w, dimensionality, classifier.features)};
            classifier.muse.dict.filterChiSquared(bopForWindow);
            bagTest = mergeBobs(bagTest, bopForWindow);
        }

        return initLibLinear(bagTest, classifier.muse.dict)[0];
    }

    private MUSE.BagOfBigrams[] fitOneWindow(
            Instances samples,
            int[] windowLengths, boolean mean,
            HistogramType histType,
            MUSE model,
            int[][] word, int f, int dimensionality, int w) {
        MUSE modelForWindow = new MUSE(f, maxS, histType, windowLengths, mean);

        MUSE.BagOfBigrams[] bopForWindow = modelForWindow.createBagOfPatterns(word, samples, w, dimensionality, f);
        modelForWindow.trainChiSquared(bopForWindow, chi);

        model.dict.dictChi.putAll(modelForWindow.dict.dictChi);

        return bopForWindow;
    }

    private MUSE.BagOfBigrams[] mergeBobs(
            MUSE.BagOfBigrams[] bop,
            MUSE.BagOfBigrams[] bopForWindow) {
        if (bop == null) {
            bop = bopForWindow;
        } else {
            for (int i = 0; i < bop.length; i++) {
                bop[i].bob.putAll(bopForWindow[i].bob);
            }
        }
        return bop;
    }

    public static Problem initLibLinearProblem(
            final MUSE.BagOfBigrams[] bob, final MUSE.Dictionary dict, final double bias) {
        Linear.resetRandom();

        Problem problem = new Problem();
        problem.bias = bias;
        problem.y = getLabels(bob);

        final FeatureNode[][] features = initLibLinear(bob, dict);

        problem.n = dict.size() + 1;
        problem.l = features.length;
        problem.x = features;
        return problem;
    }

    public static double[] getLabels(final MUSE.BagOfBigrams[] bagOfPatternsTestSamples) {
        double[] labels = new double[bagOfPatternsTestSamples.length];
        for (int i = 0; i < bagOfPatternsTestSamples.length; i++) {
            labels[i] = bagOfPatternsTestSamples[i].label;
        }
        return labels;
    }

    protected static FeatureNode[][] initLibLinear(
            final MUSE.BagOfBigrams[] bob,
            final MUSE.Dictionary dict) {

        FeatureNode[][] featuresTrain = new FeatureNode[bob.length][];
        for (int j = 0; j < bob.length; j++) {
            MUSE.BagOfBigrams bop = bob[j];
            ArrayList<FeatureNode> features = new ArrayList<FeatureNode>(bop.bob.size());
            for (ObjectIntCursor<MUSE.MuseWord> word : bop.bob) {
                if (word.value > 0 ) {
                    features.add(new FeatureNode(dict.getWordChi(word.key), word.value));
                }
            }

            FeatureNode[] featuresArray = features.toArray(new FeatureNode[]{});
            Arrays.sort(featuresArray, new Comparator<FeatureNode>() {
                public int compare(FeatureNode o1, FeatureNode o2) {
                    return Integer.compare(o1.index, o2.index);
                }
            });

            featuresTrain[j] = featuresArray;
        }
        return featuresTrain;
    }

    @SuppressWarnings("static-access")
    protected static int trainLibLinear(
            final Problem prob, final SolverType solverType, double c,
            int iter, double p, int nr_fold) {
        final Parameter param = new Parameter(solverType, c, iter, p);

        ThreadLocal<Random> myRandom = new ThreadLocal<>();
        myRandom.set(new Random(1));
        Random random = myRandom.get();

        int k;
        final int l = prob.l;
        final int[] perm = new int[l];

        if (nr_fold > l) {
            nr_fold = l;
        }
        final int[] fold_start = new int[nr_fold + 1];

        for (k = 0; k < l; k++) {
            perm[k] = k;
        }
        for (k = 0; k < l; k++) {
            int j = k + random.nextInt(l - k);
            swap(perm, k, j);
        }
        for (k = 0; k <= nr_fold; k++) {
            fold_start[k] = k * l / nr_fold;
        }

        final AtomicInteger correct = new AtomicInteger(0);

        final int fold = nr_fold;
        Linear myLinear = new Linear();
        myLinear.disableDebugOutput();
        myLinear.resetRandom(); // reset random component of liblinear for reproducibility

        for (int i = 0; i < fold; i++) {
            int begin = fold_start[i];
            int end = fold_start[i + 1];
            int j, kk;
            Problem subprob = new Problem();

            subprob.bias = prob.bias;
            subprob.n = prob.n;
            subprob.l = l - (end - begin);
            subprob.x = new Feature[subprob.l][];
            subprob.y = new double[subprob.l];

            kk = 0;
            for (j = 0; j < begin; j++) {
                subprob.x[kk] = prob.x[perm[j]];
                subprob.y[kk] = prob.y[perm[j]];
                ++kk;
            }
            for (j = end; j < l; j++) {
                subprob.x[kk] = prob.x[perm[j]];
                subprob.y[kk] = prob.y[perm[j]];
                ++kk;
            }

            de.bwaldvogel.liblinear.Model submodel = myLinear.train(subprob, param);
            for (j = begin; j < end; j++) {
                correct.addAndGet(prob.y[perm[j]] == myLinear.predict(submodel, prob.x[perm[j]]) ? 1 : 0);
            }
        }
        return correct.get();
    }

    private static void swap(int[] array, int idxA, int idxB) {
        int temp = array[idxA];
        array[idxA] = array[idxB];
        array[idxB] = temp;
    }

    public int[] getWindowLengths(final Instances samples, boolean norm) {
        int min = norm && MIN_WINDOW_LENGTH<=2? Math.max(3,MIN_WINDOW_LENGTH) : MIN_WINDOW_LENGTH;
        int max = Math.min(channelLength(samples), MAX_WINDOW_LENGTH);
        int[] wLengths = new int[max - min + 1];
        for (int w = min, a = 0; w <= max; w++, a++) {
            wLengths[a] = w;
        }
        return wLengths;
    }

    protected static int binlog(int bits) {
        int log = 0;
        if ((bits & 0xffff0000) != 0) {
            bits >>>= 16;
            log = 16;
        }
        if (bits >= 256) {
            bits >>>= 8;
            log += 8;
        }
        if (bits >= 16) {
            bits >>>= 4;
            log += 4;
        }
        if (bits >= 4) {
            bits >>>= 2;
            log += 2;
        }
        return log + (bits >>> 1);
    }

    protected static int instanceLength(Instance inst) {
        int length = inst.numAttributes();
        if (inst.classIndex() >= 0)
            --length;

        return length;
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

    public static class MUSEModel {

        public MUSEModel() {
        }

        public MUSEModel(
                boolean normed,
                int features,
                HistogramType histType,
                MUSE model,
                de.bwaldvogel.liblinear.Model linearModel
        ) {
            this.normed = normed;
            this.features = features;
            this.muse = model;
            this.linearModel = linearModel;
            this.histType = histType;
        }
        public boolean normed;

        // the best number of Fourier values to be used
        public int features;

        // the trained MUSE transformation
        public MUSE muse;

        // the trained liblinear classifier
        public de.bwaldvogel.liblinear.Model linearModel;

        public HistogramType histType;
    }

    /**
     * The WEASEL+MUSE-Model as published in
     *
     * Schäfer, P., Leser, U.: Multivariate Time Series Classification
     * with WEASEL+MUSE. arXiv 2017
     * http://arxiv.org/abs/1711.11343
     */
    public static class MUSE {

        public int alphabetSize;
        public int maxF;
        public HistogramType histogramType = null;

        public int[] windowLengths;
        public boolean normMean;
        public SFA[][] signature;
        public Dictionary dict;

        public static class MuseWord {
            int w = 0;
            int dim = 0;
            int word = 0;
            int word2 = 0;

            public MuseWord(int w, int dim, int word, int word2) {
                this.w = w;
                this.dim = dim;
                this.word = word;
                this.word2 = word2;
            }

            @Override
            public boolean equals(Object o) {
                if (this == o) return true;
                MuseWord museWord = (MuseWord) o;
                return w == museWord.w &&
                        dim == museWord.dim &&
                        word == museWord.word &&
                        word2 == museWord.word2;
            }

            @Override
            public int hashCode() {
                int result = 1;
                result = 31 * result + Integer.hashCode(word);
                result = 31 * result + Integer.hashCode(word2);
                result = 31 * result + Integer.hashCode(w);
                result = 31 * result + Integer.hashCode(dim);
                return result;
            }

            @Override
            public String toString() {
                return w + "-" + dim + "-" + word + "-" + word2;
            }
        }

        /**
         * The WEASEL-model: a histogram of SFA word and bi-gram frequencies
         */
        public static class BagOfBigrams {
            public ObjectIntHashMap<MuseWord> bob;
            public Double label;

            public BagOfBigrams(int size, Double label) {
                this.bob = new ObjectIntHashMap<>(size);
                this.label = label;
            }
        }

        /**
         * A dictionary that maps each SFA word to an integer.
         *
         * Condenses the SFA word space.
         */
        public static class Dictionary {
            public ObjectIntHashMap<MuseWord> dictChi;
            public ArrayList<MuseWord> inverseDict;

            public Dictionary() {
                this.dictChi = new ObjectIntHashMap<>();
                this.inverseDict = new ArrayList<>();
                this.inverseDict.add(new MuseWord(0, 0, 0, 0));
            }

            public void reset() {
                this.dictChi = new ObjectIntHashMap<>();
                this.inverseDict = new ArrayList<>();
                this.inverseDict.add(new MuseWord(0, 0, 0, 0));
            }

            public int getWordChi(MuseWord word) {
                int index = 0;
                if ((index = this.dictChi.indexOf(word)) > -1) {
                    return this.dictChi.indexGet(index);
                } else {
                    int newWord = this.dictChi.size() + 1;
                    this.dictChi.put(word, newWord);
                    inverseDict.add(/*newWord,*/ word);
                    return newWord;
                }
            }

            public int size() {
                return this.dictChi.size();
            }

            public void filterChiSquared(final BagOfBigrams[] bagOfPatterns) {
                for (int j = 0; j < bagOfPatterns.length; j++) {
                    ObjectIntHashMap<MuseWord> oldMap = bagOfPatterns[j].bob;
                    bagOfPatterns[j].bob = new ObjectIntHashMap<>();
                    for (ObjectIntCursor<MuseWord> word : oldMap) {
                        if (this.dictChi.containsKey(word.key) && word.value > 0) {
                            bagOfPatterns[j].bob.put(word.key, word.value);
                        }
                    }
                }
            }
        }

        /**
         * Create a WEASEL+MUSE model.
         *
         * @param maxF          Length of the SFA words
         * @param maxS          alphabet size
         * @param histogramType histogram types (EQUI-Depth and/or EQUI-Frequency) to use
         * @param windowLengths the set of window lengths to use for extracting SFA words from
         *                      time series.
         * @param normMean      set to true, if mean should be set to 0 for a window
         */
        public MUSE(
                int maxF,
                int maxS,
                HistogramType histogramType,
                int[] windowLengths,
                boolean normMean) {
            this.maxF = maxF + maxF % 2; // even number
            this.alphabetSize = maxS;
            this.windowLengths = windowLengths;
            this.normMean = normMean;
            this.dict = new Dictionary();
            this.signature = new SFA[windowLengths.length][];
            this.histogramType = histogramType;
        }

        /**
         * Create SFA words and bigrams for all samples
         *
         * @param samples
         * @return
         */
        protected int[][] createWords(final Instances samples, final int index) {

            // SFA quantization
            if (this.signature[index] == null) {
                this.signature[index] = new SFA[numDimensions(samples)];
                for (int i = 0; i < this.signature[index].length; i++) {
                    this.signature[index][i] = new SFA(this.histogramType);
                    this.signature[index][i].fitWindowing(samples, this.windowLengths[index], this.maxF,
                            this.alphabetSize, this.normMean, i);
                }
            }

            // create words
            Instances[] split = splitMultivariateInstances(samples);
            final int[][] words = new int[samples.numInstances() * split.length][];
            int pos = 0;
            for (int i = 0; i < samples.numInstances(); i++) {
                for (int n = 0; n < split.length; n++) {
                    if (channelLength(samples) >= this.windowLengths[index]) {
                        words[pos] = this.signature[index][n].transformWindowingInt(split[n].get(i), this.maxF);
                    } else {
                        words[pos] = new int[]{};
                    }
                    pos++;
                }
            }

            return words;
        }

        /**
         * Create SFA words and bigrams for a single sample
         *
         * @param sample
         * @return
         */
        private int[][] createWords(final Instance sample, final int index) {
            // create words
            Instance[] split = splitMultivariateInstance(sample);
            final int[][] words = new int[split.length][];
            for (int n = 0; n < split.length; n++) {
                if (channelLength(sample) >= this.windowLengths[index]) {
                    words[n] = this.signature[index][n].transformWindowingInt(split[n], this.maxF);
                } else {
                    words[n] = new int[]{};
                }
            }

            return words;
        }

        /**
         * Implementation based on:
         * https://github.com/scikit-learn/scikit-learn/blob/c957249/sklearn/feature_selection/univariate_selection.py#L170
         */
        public void trainChiSquared(final BagOfBigrams[] bob, double chi_limit) {
            // Chi2 Test
            ObjectIntHashMap<MuseWord> featureCount = new ObjectIntHashMap<>(bob[0].bob.size());
            LongDoubleHashMap classProb = new LongDoubleHashMap(10);
            LongObjectHashMap<ObjectIntHashMap<MuseWord>> observed = new LongObjectHashMap<>(bob[0].bob.size());

            // count number of samples with this word
            for (BagOfBigrams bagOfPattern : bob) {
                long label = bagOfPattern.label.longValue();
                if (!observed.containsKey(label)) {
                    observed.put(label, new ObjectIntHashMap<>());
                }
                for (ObjectIntCursor<MuseWord> word : bagOfPattern.bob) {
                    if (word.value > 0) {
                        featureCount.putOrAdd(word.key, 1, 1);
                        observed.get(label).putOrAdd(word.key, 1, 1);
                    }
                }
            }

            // samples per class
            for (BagOfBigrams bagOfPattern : bob) {
                long label = bagOfPattern.label.longValue();
                classProb.putOrAdd(label, 1, 1);
            }

            // chi-squared: observed minus expected occurrence
            ObjectHashSet<MuseWord> chiSquare = new ObjectHashSet<>(featureCount.size());
            for (LongDoubleCursor classLabel : classProb) {
                classLabel.value /= bob.length;
                if (observed.get(classLabel.key) != null) {
                    ObjectIntHashMap<MuseWord> observe = observed.get(classLabel.key);
                    for (ObjectIntCursor<MuseWord> feature : featureCount) {
                        double expected = classLabel.value * feature.value;
                        double chi = observe.get(feature.key) - expected;
                        double newChi = chi * chi / expected;
                        if (newChi >= chi_limit
                                && !chiSquare.contains(feature.key)) {
                            chiSquare.add(feature.key);
                        }
                    }
                }
            }

            // best elements above limit
            for (int j = 0; j < bob.length; j++) {
                for (ObjectIntCursor<MuseWord> cursor : bob[j].bob) {
                    if (!chiSquare.contains(cursor.key)) {
                        bob[j].bob.values[cursor.index] = 0;
                    }
                }
            }
        }

        /**
         * Create words and bi-grams for all window lengths
         */
        public BagOfBigrams createBagOfPatterns(
                final int[][] words,
                final Instance sample,
                final int w,    // index of used windowSize
                final int dimensionality,
                final int wordLength) {
            final byte usedBits = (byte) binlog(this.alphabetSize);
            final int mask = (1 << (usedBits * wordLength)) - 1;

            BagOfBigrams bop = new BagOfBigrams(100, sample.classValue());

            // create subsequences
            if (this.windowLengths[w] >= wordLength) {
                for (int dim = 0; dim < dimensionality; dim++) {
                    for (int offset = 0; offset < words[dim].length; offset++) {
                        MuseWord word = new MuseWord(w, dim, words[dim][offset] & mask, 0);
                        //int dict = this.dict.getWord(word);
                        bop.bob.putOrAdd(word, 1, 1);

                        // add bigrams
                        if (this.windowLengths[this.windowLengths.length-1] < 200 // avoid for too large datasets
                                //&& useBigrams
                                && (offset - this.windowLengths[w] >= 0)) {
                            MuseWord bigram = new MuseWord(w, dim,
                                    (words[dim][offset - this.windowLengths[w]] & mask),
                                    words[dim][offset] & mask);
                            //int newWord = this.dict.getWord(bigram);
                            bop.bob.putOrAdd(bigram, 1, 1);
                        }
                    }
                }
            }

            return bop;
        }

        /**
         * Create words and bi-grams for all window lengths
         */
        public BagOfBigrams[] createBagOfPatterns(
                final int[][] wordsForWindowLength,
                final Instances samples,
                final int w,    // index of used windowSize
                final int dimensionality,
                final int wordLength) {
            List<BagOfBigrams> bagOfPatterns = new ArrayList<>(
                    samples.numInstances() * dimensionality);

            final byte usedBits = (byte) binlog(this.alphabetSize);
            final int mask = (1 << (usedBits * wordLength)) - 1;

            // iterate all samples and create a muse model for each
            for (int i = 0, j = 0; i < samples.numInstances(); i++, j += dimensionality) {
                BagOfBigrams bop = new BagOfBigrams(100, samples.get(i).classValue());

                // create subsequences
                if (this.windowLengths[w] >= wordLength) {
                    for (int dim = 0; dim < dimensionality; dim++) {
                        for (int offset = 0; offset < wordsForWindowLength[j + dim].length; offset++) {
                            MuseWord word = new MuseWord(w, dim, wordsForWindowLength[j + dim][offset] & mask, 0);
                            //int dict = this.dict.getWord(word);
                            bop.bob.putOrAdd(word, 1, 1);

                            // add bigrams
                            if (this.windowLengths[this.windowLengths.length-1] < 200 // avoid for too large datasets
                                    //&& useBigrams
                                    && (offset - this.windowLengths[w] >= 0)) {
                                MuseWord bigram = new MuseWord(w, dim,
                                        (wordsForWindowLength[j + dim][offset - this.windowLengths[w]] & mask),
                                        wordsForWindowLength[j + dim][offset] & mask);
                                //int newWord = this.dict.getWord(bigram);

                                bop.bob.putOrAdd(bigram, 1, 1);
                            }
                        }
                    }
                }
                bagOfPatterns.add(bop);
            }

            return bagOfPatterns.toArray(new BagOfBigrams[]{});
        }
    }

    /**
     * SFA using the ANOVA F-statistic to determine the best Fourier coefficients
     * (those that best separate between class labels) as opposed to using the first
     * ones.
     */
    public static class SFA {
        // distribution of Fourier values
        public transient ArrayList<Double>[] orderLine;

        public HistogramType histogramType = HistogramType.EQUI_DEPTH;

        public int alphabetSize = 256;
        public byte neededBits = (byte) binlog(this.alphabetSize);
        public int wordLength = 0;
        public boolean initialized = false;

        public int maxWordLength;

        // The Momentary Fourier Transform
        public MFT transformation;

        // use binning / bucketing
        public double[][] bins;

        public SFA(HistogramType histogramType){
            this.histogramType = histogramType;
        }

        @SuppressWarnings("unchecked")
        private void init(int l, int alphabetSize) {
            this.wordLength = l;
            this.maxWordLength = l;
            this.alphabetSize = alphabetSize;
            this.initialized = true;

            // l-dimensional bins
            this.alphabetSize = alphabetSize;
            this.neededBits = (byte) binlog(alphabetSize);

            this.bins = new double[l][alphabetSize - 1];
            for (double[] row : this.bins) {
                Arrays.fill(row, Double.MAX_VALUE);
            }

            this.orderLine = new ArrayList[l];
            for (int i = 0; i < this.orderLine.length; i++) {
                this.orderLine[i] = new ArrayList<>();
            }
        }

        /**
         * Extracts sliding windows from the multivariate time series and
         * trains SFA based on the sliding windows.
         * At the end of this call, the quantization bins are set.
         *
         * @param timeSeries   A set of multivariate sample time series
         * @param windowLength The queryLength of each sliding window
         * @param wordLength   the SFA word-queryLength
         * @param symbols      the SFA alphabet size
         * @param normMean     if set, the mean is subtracted from each sliding window
         * @param dim          the dimension of the multivariate time series to use
         */
        public void fitWindowing(Instances timeSeries, int windowLength, int wordLength, int symbols, boolean normMean, int dim) {
            ArrayList<double[]> sa = new ArrayList<>(timeSeries.numInstances() * numDimensions(timeSeries) *
                    channelLength(timeSeries) / windowLength);

            for (Instance t : timeSeries) {
                Collections.addAll(sa, getDisjointSequences(t.relationalValue(0).get(dim), windowLength, normMean));
            }

            double[][] allSamples = new double[sa.size()][];
            for (int i = 0; i < sa.size(); i++) {
                allSamples[i] = sa.get(i);
            }

            fitTransform(allSamples, wordLength, symbols, normMean);
        }

        /**
         * Extracts disjoint subsequences
         */
        public double[][] getDisjointSequences(Instance t, int windowSize, boolean normMean) {
            // extract subsequences
            int amount = instanceLength(t) / windowSize;
            double[][] subsequences = new double[amount][windowSize];

            double[] data = toArrayNoClass(t);
            for (int i = 0; i < amount; i++) {
                double[] subsequenceData = new double[windowSize];
                System.arraycopy(data, i * windowSize, subsequenceData, 0, windowSize);
                //TODO weird norm bit from SFA code, calls normalisation function but doesnt actually normalise.
                subsequences[i] = subsequenceData; //z_norm(subsequenceData, normMean);
            }

            return subsequences;
        }

        public double[] z_norm(double[] data, boolean normMean) {
            double mean = 0.0;
            double stddev = 0;

            // get mean +stddev values
            double var = 0;
            for (double value : data) {
                mean += value;
                var += value * value;

            }
            mean /= data.length;

            double norm = 1.0 / ((double) data.length);
            double buf = norm * var - mean * mean;
            if (buf > 0) {
                stddev = Math.sqrt(buf);
            }

            double inverseStddev = (stddev != 0) ? 1.0 / stddev : 1.0;
            if (normMean) {
                for (int i = 0; i < data.length; i++) {
                    data[i] = (data[i] - mean) * inverseStddev;
                }
            } else if (inverseStddev != 1.0) {
                for (int i = 0; i < data.length; i++) {
                    data[i] *= inverseStddev;
                }
            }
            return data;
        }

        public void fitTransform(double[][] samples, int wordLength, int symbols, boolean normMean) {
            if (!this.initialized) {
                init(wordLength, symbols);

                if (this.transformation == null) {
                    this.transformation = new MFT(samples[0].length, normMean);
                }
            }

            fillOrderline(samples, wordLength);

            if (this.histogramType == HistogramType.EQUI_DEPTH) {
                divideEquiDepthHistogram();
            } else if (this.histogramType == HistogramType.EQUI_FREQUENCY) {
                divideEquiWidthHistogram();
            }

            this.orderLine = null;;
        }

        /**
         * Use equi-width binning to divide the orderline
         */
        protected void divideEquiWidthHistogram() {
            int i = 0;
            for (List<Double> elements : this.orderLine) {
                if (!elements.isEmpty()) {
                    // apply the split
                    double first = elements.get(0);
                    double last = elements.get(elements.size() - 1);
                    double intervalWidth = (last - first) / (this.alphabetSize);

                    for (int c = 0; c < this.alphabetSize - 1; c++) {
                        this.bins[i][c] = intervalWidth * (c + 1) + first;
                    }
                }
                i++;
            }
        }

        /**
         * Use equi-depth binning to divide the orderline
         */
        protected void divideEquiDepthHistogram() {
            // For each real and imaginary part
            for (int i = 0; i < this.bins.length; i++) {
                // Divide into equi-depth intervals
                double depth = this.orderLine[i].size() / (double) (this.alphabetSize);

                int pos = 0;
                long count = 0;
                for (Double value : this.orderLine[i]) {
                    if (++count > Math.ceil(depth * (pos + 1))
                            && (pos == 0 || this.bins[i][pos - 1] != value)) {
                        this.bins[i][pos++] = value;
                    }
                }
            }
        }

        /**
         * Fills data in the orderline
         *
         * @param samples A set of samples
         */
        protected void fillOrderline(double[][] samples, int l) {
            double[][] transformedSamples = new double[samples.length][];

            for (int i = 0; i < samples.length; i++) {
                // approximation
                transformedSamples[i] = this.transformation.transform(samples[i], l);

                for (int j = 0; j < transformedSamples[i].length; j++) {
                    // round to 2 decimal places to reduce noise
                    double value = Math.round(transformedSamples[i][j] * 100.0) / 100.0;
                    this.orderLine[j].add(value);
                }
            }

            // Sort ascending by value
            for (List<Double> element : this.orderLine) {
                Collections.sort(element);
            }
        }


        /**
         * Quantization of a DFT approximation to its SFA word
         *
         * @param approximation the DFT approximation of a time series
         * @return
         */
        public short[] quantization(double[] approximation) {
            int i = 0;
            short[] word = new short[approximation.length];
            for (double value : approximation) {
                // lookup character:
                short c = 0;
                for (; c < this.bins[i].length; c++) {
                    if (value < this.bins[i][c]) {
                        break;
                    }
                }
                word[i++] = c;
            }
            return word;
        }

        /**
         * Transforms a single time series to its SFA word
         *
         * @param timeSeries a sample
         * @param approximation the DFT approximation, if available, else pass 'null'
         * @return
         */
        public short[] transform(double[] timeSeries, double[] approximation) {
            if (!this.initialized) {
                throw new RuntimeException("Please call fitTransform() first.");
            }
            if (approximation == null) {
                // get approximation of the time series
                approximation = this.transformation.transform(timeSeries, this.maxWordLength);
            }

            // use lookup table (bins) to get the word from the approximation
            return quantization(approximation);
        }


        /**
         * Transforms a set of time series to SFA words.
         *
         * @param samples a set of samples
         * @param approximation the DFT approximations, if available, else pass 'null'
         * @return
         */
        public short[][] transform(double[][] samples, double[][] approximation) {
            if (!this.initialized) {
                throw new RuntimeException("Please call fitTransform() first.");
            }
            short[][] transform = new short[samples.length][];
            for (int i = 0; i < transform.length; i++) {
                transform[i] = transform(samples[i], approximation[i]);
            }

            return transform;
        }

        /**
         * Returns a long containing the values in bytes.
         */
        protected static long fromByteArrayOne(short[] bytes, int to, byte usedBits) {
            int shortsPerLong = 60 / usedBits;
            to = Math.min(bytes.length, to);

            long bits = 0;
            int start = 0;
            long shiftOffset = 1;
            for (int i = start, end = Math.min(to, shortsPerLong + start); i < end; i++) {
                for (int j = 0, shift = 1; j < usedBits; j++, shift <<= 1) {
                    if ((bytes[i] & shift) != 0) {
                        bits |= shiftOffset;
                    }
                    shiftOffset <<= 1;
                }
            }

            return bits;
        }

        protected static long createWord(short[] words, int features, byte usedBits) {
            return fromByteArrayOne(words, features, usedBits);
        }

        public int[] transformWindowingInt(Instance ts, int wordLength) {
            short[][] words = transformWindowing(ts);
            int[] intWords = new int[words.length];
            for (int i = 0; i < words.length; i++) {
                intWords[i] = (int) createWord(words[i], wordLength, this.neededBits);
            }
            return intWords;
        }

        /**
         * Extracts sliding windows from a time series and transforms it to its SFA
         * word.
         * <p>
         * Returns the SFA words as short[] (from Fourier transformed windows). Each
         * short corresponds to one character.
         *
         * @param timeSeries a sample
         * @return
         */
        public short[][] transformWindowing(Instance timeSeries) {
            double[][] mft = this.transformation.transformWindowing(timeSeries, this.maxWordLength);

            short[][] words = new short[mft.length][];
            for (int i = 0; i < mft.length; i++) {
                words[i] = quantization(mft[i]);
            }

            return words;
        }
    }

    /**
     * The Momentary Fourier Transform is alternative algorithm of
     * the Discrete Fourier Transform for overlapping windows. It has
     * a constant computational complexity for in the window queryLength n as
     * opposed to O(n log n) for the Fast Fourier Transform algorithm.
     * <p>
     * It was first published in:
     * Albrecht, S., Cumming, I., Dudas, J.: The momentary fourier transformation
     * derived from recursive matrix transformations. In: Digital Signal Processing
     * Proceedings, 1997., IEEE (1997)
     *
     */
    public static class MFT implements Serializable {
        private static final long serialVersionUID = 8508604292241736378L;

        private int windowSize = 0;
        private int startOffset = 0;
        private transient DoubleFFT_1D fft = null;

        public MFT(int windowSize, boolean normMean) {
            this.windowSize = windowSize;

            this.fft = new DoubleFFT_1D(this.windowSize);

            // ignore DC value?
            this.startOffset = normMean ? 2 : 0;
        }

        public double[] transform(double[] series, int wordLength) {
            double[] data = new double[this.windowSize];
            System.arraycopy(series, 0, data, 0, Math.min(this.windowSize, series.length));
            this.fft.realForward(data);
            data[1] = 0; // DC-coefficient imaginary part

            // make it even length for uneven windowSize
            double[] copy = new double[wordLength];
            int length = Math.min(this.windowSize - this.startOffset, wordLength);
            System.arraycopy(data, this.startOffset, copy, 0, length);

            // norming
            int sign = 1;
            for (int i = 0; i < copy.length; i++) {
                copy[i] *= sign;
                sign *= -1;
            }

            return copy;
        }

        /**
         * Transforms a time series, extracting windows and using *momentary* fourier
         * transform for each window. Results in one Fourier transform for each
         * window. Returns only the first l/2 Fourier coefficients for each window.
         *
         * @param timeSeries the time series to be transformed
         * @param l          the number of Fourier values to use (equal to l/2 Fourier
         *                   coefficients). If l is uneven, l+1 Fourier values are returned. If
         *                   windowSize is smaller than l, only the first windowSize Fourier
         *                   values are set.
         * @return returns only the first l/2 Fourier coefficients for each window.
         */
        public double[][] transformWindowing(Instance timeSeries, int l) {
            int wordLength = Math.min(windowSize, l + this.startOffset);
            wordLength += wordLength%2; // make it even
            double[] phis = new double[wordLength];

            for (int u = 0; u < phis.length; u += 2) {
                double uHalve = -u / 2;
                phis[u] = realPartEPhi(uHalve, this.windowSize);
                phis[u + 1] = complexPartEPhi(uHalve, this.windowSize);
            }

            // means and stddev for each sliding window
            int end = Math.max(1, instanceLength(timeSeries) - this.windowSize + 1);
            double[] means = new double[end];
            double[] stds = new double[end];
            calcIncrementalMeanStddev(this.windowSize, toArrayNoClass(timeSeries), means, stds);

            double[][] transformed = new double[end][];

            // holds the DFT of each sliding window
            double[] mftData = new double[wordLength];
            double[] data = toArrayNoClass(timeSeries);

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
                    System.arraycopy(toArrayNoClass(timeSeries), 0, dft, 0, Math.min(this.windowSize, data.length));

                    this.fft.realForward(dft);
                    dft[1] = 0; // DC-coefficient imag part

                    // if windowSize > mftData.queryLength, the remaining data should be 0 now.
                    System.arraycopy(dft, 0, mftData, 0, Math.min(mftData.length, dft.length));
                }

                // normalization for lower bounding
                double[] copy = new double[l];
                System.arraycopy(mftData, this.startOffset, copy, 0, Math.min(l, mftData.length-this.startOffset));

                transformed[t] = normalizeFT(copy, stds[t]);
            }

            return transformed;
        }

        /**
         * Gets the means and stddevs for all sliding windows of a time series
         */
        public void calcIncrementalMeanStddev(
                int windowLength,
                double[] tsData,
                double[] means,
                double[] stds) {
            double sum = 0;
            double squareSum = 0;

            // it is faster to multiply than to divide
            double rWindowLength = 1.0 / (double) windowLength;

            for (int ww = 0; ww < Math.min(tsData.length, windowLength); ww++) {
                sum += tsData[ww];
                squareSum += tsData[ww] * tsData[ww];
            }

            // first window
            means[0] = sum * rWindowLength;
            double buf = squareSum * rWindowLength - means[0] * means[0];
            stds[0] = buf > 0 ? Math.sqrt(buf) : 0;

            // remaining windows
            for (int w = 1, end = tsData.length - windowLength + 1; w < end; w++) {
                sum += tsData[w + windowLength - 1] - tsData[w - 1];
                means[w] = sum * rWindowLength;

                squareSum += tsData[w + windowLength - 1] * tsData[w + windowLength - 1] - tsData[w - 1] * tsData[w - 1];
                buf = squareSum * rWindowLength - means[w] * means[w];
                stds[w] = buf > 0 ? Math.sqrt(buf) : 0;
            }
        }

        /**
         * Calculate the real part of a multiplication of two complex numbers
         */
        private double complexMultiplyRealPart(double r1, double im1, double r2, double im2) {
            return r1 * r2 - im1 * im2;
        }

        /**
         * Caluculate the imaginary part of a multiplication of two complex numbers
         */
        private double complexMultiplyImagPart(double r1, double im1, double r2, double im2) {
            return r1 * im2 + r2 * im1;
        }

        /**
         * Real part of e^(2*pi*u/M)
         */
        private double realPartEPhi(double u, double M) {
            return Math.cos(2 * Math.PI * u / M);
        }

        /**
         * Imaginary part of e^(2*pi*u/M)
         */
        private double complexPartEPhi(double u, double M) {
            return -Math.sin(2 * Math.PI * u / M);
        }

        /**
         * Apply normalization to the Fourier coefficients to allow lower bounding in Euclidean space
         */
        private double[] normalizeFT(double[] copy, double std) {
            double normalisingFactor = 1.0 ;//std > 0 ? 1.0 / std : 1.0; TODO another weird norm bit from SFA code
            int sign = 1;
            for (int i = 0; i < copy.length; i++) {
                copy[i] *= sign * normalisingFactor;
                sign *= -1;
            }
            return copy;
        }
    }

    public static void main(String[] args) throws Exception{
        int fold =0;

        //Minimum working example
        String dataset = "RacketSports";
        Instances train = DatasetLoading.loadDataNullable("E:\\Datasets\\Multivariate_arff\\"+dataset+"\\"+dataset+"_TRAIN.arff");
        Instances test = DatasetLoading.loadDataNullable("E:\\Datasets\\Multivariate_arff\\"+dataset+"\\"+dataset+"_TEST.arff");
        Instances[] data = resampleMultivariateTrainAndTestInstances(train, test, fold);
        train = data[0];
        test = data[1];

        WEASEL_MUSE c;
        double accuracy;

        c = new WEASEL_MUSE();
        c.setSeed(fold);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("WEASEL_MUSE accuracy on " + dataset + " fold " + fold + " = " + accuracy);
    }
}
