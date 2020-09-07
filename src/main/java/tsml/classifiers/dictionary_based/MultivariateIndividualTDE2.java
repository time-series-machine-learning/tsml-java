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

import com.carrotsearch.hppc.DoubleDoubleHashMap;
import com.carrotsearch.hppc.DoubleObjectHashMap;
import com.carrotsearch.hppc.ObjectHashSet;
import com.carrotsearch.hppc.ObjectIntHashMap;
import com.carrotsearch.hppc.cursors.DoubleDoubleCursor;
import com.carrotsearch.hppc.cursors.ObjectIntCursor;
import evaluation.storage.ClassifierResults;
import tsml.classifiers.dictionary_based.bitword.BitWord;
import tsml.classifiers.dictionary_based.bitword.BitWordInt;
import tsml.classifiers.dictionary_based.bitword.BitWordLong;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;

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
 *
 * @author Matthew Middlehurst
 */
public class MultivariateIndividualTDE2 extends IndividualTDE {

    //all sfa words found in original buildClassifier(), no numerosity reduction/shortening applied
    private BitWord[/*instance*/][/*dimension*/][/*windowindex*/] SFAwords;

    //histograms of words of the current wordlength with numerosity reduction applied (if selected)
    private ArrayList<BagMV> bags;

    //breakpoints to be found by MCB or IGB
    private double[/*dimension*/][/*letterindex*/][/*breakpointsforletter*/] breakpoints;

    private static final long serialVersionUID = 2L;

    public MultivariateIndividualTDE2(int wordLength, int alphabetSize, int windowSize, boolean normalise, int levels,
                                      boolean IGB, boolean multiThread, int numThreads, ExecutorService ex) {
        super(wordLength, alphabetSize, windowSize, normalise, levels, IGB, multiThread, numThreads, ex);
    }

    public MultivariateIndividualTDE2(int wordLength, int alphabetSize, int windowSize, boolean normalise, int levels,
                                      boolean IGB) {
        super(wordLength, alphabetSize, windowSize, normalise, levels, IGB);
    }

    /**
     * Used when shortening histograms, copies 'meta' data over, but with shorter
     * word length, actual shortening happens separately
     */
    public MultivariateIndividualTDE2(MultivariateIndividualTDE2 boss, int wordLength) {
        super(boss, wordLength);

        this.SFAwords = boss.SFAwords;
        this.bags = new ArrayList<>(boss.bags.size());
        this.breakpoints = boss.breakpoints;
    }

    //map of <word, level, dimension> => count
    public static class BagMV extends HashMap<Word, Integer> implements Serializable {
        private int classVal;

        public BagMV() {
            super();
        }

        public BagMV(int classValue) {
            super();
            classVal = classValue;
        }

        public int getClassVal() { return classVal; }
        public void setClassVal(int classVal) { this.classVal = classVal; }
    }

    public static class Word implements Serializable {
        BitWord[] word;
        byte level;

        public Word(BitWord[] word, byte level) {
            this.word = word;
            this.level = level;
        }

        @Override
        public boolean equals(Object o) {
            if (o instanceof Word) {
                return Arrays.equals(word, ((Word) o).word) && level == ((Word) o).level;
            }
            return false;
        }

        @Override
        public int hashCode() {
            int result = 1;
            result = 31 * result + Arrays.hashCode(word);
            result = 31 * result + Byte.hashCode(level);
            return result;
        }

        @Override
        public String toString(){
            return "[" + Arrays.toString(word) + "," + level + "]";
        }
    }

    public ArrayList<BagMV> getMultivariateBags() { return bags; }

    @Override
    public void clean() {
        SFAwords = null;
    }

    /**
     * Builds a brand new boss bag from the passed fourier transformed data, rather than from
     * looking up existing transforms from earlier builds (i.e. SFAWords).
     *
     * to be used e.g to transform new test instances
     */
    private void addToSPBagSingle(BagMV bag, double[][][] dfts) {
        BitWord[] lastWord = new BitWordInt[dfts.length];
        BitWord[][] words = new BitWordInt[dfts[0].length][];

        int wInd = 0;
        int trivialMatchCount = 0;

        for (int i = 0; i < dfts[0].length; i++) {
            BitWord[] word = new BitWordInt[dfts.length];
            for (int n = 0; n < dfts.length; n++) {
                word[n] = new BitWordInt(createWord(dfts[n][i], n));
            }
            words[wInd] = word;

            if (useBigrams) {
                if (wInd - windowSize >= 0) {
                    BitWord[] bigram = new BitWordLong[dfts.length];
                    for (int n = 0; n < dfts.length; n++) {
                        bigram[n] = new BitWordLong(words[wInd - windowSize][n], word[n]);
                    }

                    Word key = new Word(bigram, (byte) -1);
                    bag.merge(key, 1, Integer::sum);
                }
            }

            //add to bag, unless num reduction applies
            if (numerosityReduction && Arrays.equals(word, lastWord)) {
                ++trivialMatchCount;
                ++wInd;
            } else {
                //if a run of equivalent words, those words essentially representing the same
                //elongated pattern. still apply numerosity reduction, however use the central
                //time position of the elongated pattern to represent its position
                addWordToPyramid(word, wInd - (trivialMatchCount / 2), bag);

                lastWord = word;
                trivialMatchCount = 0;
                ++wInd;
            }
        }
    }

    private BitWord createWord(double[] dft, int dimension) {
        BitWord word = new BitWordInt();
        for (int l = 0; l < wordLength; ++l) //for each letter
            for (int bp = 0; bp < alphabetSize; ++bp) //run through breakpoints until right one found
                if (dft[l] <= breakpoints[dimension][l][bp]) {
                    word.push(bp); //add corresponding letter to word
                    break;
                }

        return word;
    }

    /**
     * @return BOSSSpatialPyramidsTransform-ed bag, built using current parameters
     */
    private BagMV BOSSSpatialPyramidsTransform(TimeSeriesInstance inst) {
        BagMV bag = new BagMV(inst.getLabelIndex());

        double[][] split = inst.toValueArray();

        double[][][] mfts = new double[inst.getNumDimensions()][][];
        for (int d = 0; d < mfts.length; d++) {
            mfts[d] = performMFT(split[d]); //approximation
        }
        addToSPBagSingle(bag, mfts); //discretisation/bagging
        applyPyramidWeights(bag);

        return bag;
    }

    /**
     * Builds a bag from the set of words for a pre-transformed series of a given wordlength.
     */
    private void addWordsToSPBag(BagMV bag, int thisWordLength, BitWord[][] words) {
        BitWord[] lastWord = new BitWordInt[words.length];
        BitWord[][] newWords = new BitWord[words[0].length][];

        int wInd = 0;
        int trivialMatchCount = 0; //keeps track of how many words have been the same so far

        for (int i = 0; i < words[0].length; i++) {
            BitWord[] word = new BitWordInt[words.length];
            for (int n = 0; n < words.length; n++){
                word[n] = new BitWordInt(words[n][i]);

                if (wordLength != thisWordLength)
                    word[n].shorten(16-thisWordLength); //max word length, no classifier currently uses past 16.
            }
            newWords[wInd] = word;

            if (useBigrams) {
                if (wInd - windowSize >= 0) {
                    BitWord[] bigram = new BitWordLong[words.length];
                    for (int n = 0; n < words.length; n++){
                        bigram[n] = new BitWordLong(newWords[wInd - windowSize][n], word[n]);
                    }

                    Word key = new Word(bigram, (byte) -1);
                    bag.merge(key, 1, Integer::sum);
                }
            }

            //add to bag, unless num reduction applies
            if (numerosityReduction && Arrays.equals(word, lastWord)) {
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
    }

    private void applyPyramidWeights(BagMV bag) {
        for (Map.Entry<Word, Integer> ent : bag.entrySet()) {
            //find level that this quadrant is on
            int quadrant = ent.getKey().level;
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

    private void addWordToPyramid(BitWord[] word, int wInd, BagMV bag) {
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

            Word key = new Word(word, (byte)quadrant);
            bag.merge(key, 1, Integer::sum);

            qStart += numQuadrants;
        }
    }

    private BitWord[] createSFAwords(double[] inst, int dimension) {
        double[][] dfts = performMFT(inst); //approximation
        BitWord[] words = new BitWord[dfts.length];
        for (int window = 0; window < dfts.length; ++window) {
            words[window] = createWord(dfts[window], dimension);//discretisation
        }

        return words;
    }

    @Override
    public void buildClassifier(TimeSeriesInstances data) throws Exception {
        trainResults = new ClassifierResults();
        rand.setSeed(seed);
        numClasses = data.numClasses();
        trainResults.setClassifierName(getClassifierName());
        trainResults.setParas(getParameters());
        trainResults.setBuildTime(System.nanoTime());

        double[][][] split = data.toValueArray();

        breakpoints = new double[data.getMaxNumChannels()][][];
        for (int d = 0; d < breakpoints.length; d++) {
            if (IGB) breakpoints[d] = IGB(split, d, data.getClassIndexes());
            else breakpoints[d] = MCB(split, d); //breakpoints to be used for making sfa words for train
                                                 //AND test data
        }

        SFAwords = new BitWord[data.numInstances()][data.getMaxNumChannels()][];
        bags = new ArrayList<>(data.numInstances());
        rand = new Random(seed);
        seriesLength = data.getMaxLength();

        if (multiThread){

        }
        else {
            for (int inst = 0; inst < data.numInstances(); ++inst) {
                BagMV bag = new BagMV(data.get(inst).getLabelIndex());
                for (int d = 0; d < data.getMaxNumChannels(); d++) {
                    SFAwords[inst][d] = createSFAwords(split[inst][d], d);
                }
                addWordsToSPBag(bag, wordLength, SFAwords[inst]);
                applyPyramidWeights(bag);
                bags.add(bag);
            }
        }

        if (cleanAfterBuild) {
            clean();
        }
    }

    /**
     * Computes BOSS distance between two bags d(test, train), is NON-SYMETRIC operation,
     * ie d(a,b) != d(b,a).
     *
     * Quits early if the dist-so-far is greater than bestDist (assumed is in fact the dist still squared),
     * and returns Double.MAX_VALUE
     *
     * @return distance FROM instA TO instB, or Double.MAX_VALUE if it would be greater than bestDist
     */
    public double BOSSdistance(BagMV instA, BagMV instB, double bestDist) {
        double dist = 0.0;

        //find dist only from values in instA
        for (Map.Entry<Word, Integer> entry : instA.entrySet()) {
            Integer valA = entry.getValue();
            Integer valB = instB.get(entry.getKey());
            if (valB == null) valB = 1;
            dist += (valA-valB)*(valA-valB);

            if (dist > bestDist)
                return Double.MAX_VALUE;
        }

        return dist;
    }

    public double histogramIntersection(BagMV instA, BagMV instB) {
        //min vals of keys that exist in only one of the bags will always be 0
        //therefore want to only bother looking at counts of words in both bags
        //therefore will simply loop over words in a, skipping those that dont appear in b
        //no need to loop over b, since only words missed will be those not in a anyway

        double sim = 0.0;

        for (Map.Entry<Word, Integer> entry : instA.entrySet()) {
            Integer valA = entry.getValue();
            Integer valB = instB.get(entry.getKey());
            if (valB == null)
                continue;

            sim += Math.min(valA,valB);
        }

        return sim;
    }

    @Override
    public double classifyInstance(TimeSeriesInstance instance) throws Exception{
        BagMV testBag = BOSSSpatialPyramidsTransform(instance);

        //1NN distance
        double bestDist = Double.MAX_VALUE;
        double nn = 0;

        for (BagMV bag : bags) {
            double dist;
            if (histogramIntersection)
                dist = -histogramIntersection(testBag, bag);
            else dist = BOSSdistance(testBag, bag, bestDist);

            if (dist < bestDist) {
                bestDist = dist;
                nn = bag.getClassVal();
            }
        }

        return nn;
    }

    /**
     * Used within BOSSEnsemble as part of a leave-one-out crossvalidation, to skip having to rebuild
     * the classifier every time (since the n histograms would be identical each time anyway), therefore this
     * classifies the instance at the index passed while ignoring its own corresponding histogram
     *
     * @param testIndex index of instance to classify
     * @return classification
     */
    @Override
    public double classifyInstance(int testIndex) throws Exception{
        BagMV testBag = bags.get(testIndex);

        //1NN distance
        double bestDist = Double.MAX_VALUE;
        double nn = 0;

        for (int i = 0; i < bags.size(); ++i) {
            if (i == testIndex) //skip 'this' one, leave-one-out
                continue;

            double dist;
            if (histogramIntersection)
                dist = -histogramIntersection(testBag, bags.get(i));
            else dist = BOSSdistance(testBag, bags.get(i), bestDist);

            if (dist < bestDist) {
                bestDist = dist;
                nn = bags.get(i).getClassVal();
            }
        }

        return nn;
    }
}

