package timeseriesweka.classifiers;


import com.carrotsearch.hppc.*;
import com.carrotsearch.hppc.cursors.IntCursor;
import com.carrotsearch.hppc.cursors.IntIntCursor;
import com.carrotsearch.hppc.cursors.LongFloatCursor;
import de.bwaldvogel.liblinear.*;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;
import fileIO.OutFile;
import timeseriesweka.classifiers.cote.HiveCoteModule;
import utilities.*;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * WEASEL Classifier
 *
 * @author Patrick Schaefer
 *
 */
public class WEASEL extends AbstractClassifierWithTrainingData implements HiveCoteModule, TrainAccuracyEstimate {

  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
    result.setValue(TechnicalInformation.Field.AUTHOR, "P. Schaefer, U. Leser");
    result.setValue(TechnicalInformation.Field.TITLE, "Fast and Accurate Time Series Classification with WEASEL");
    result.setValue(TechnicalInformation.Field.JOURNAL, "CIKM");
    result.setValue(TechnicalInformation.Field.YEAR, "2017");

    return result;
  }

  public WEASELModel classifier;

  // WEASEL model parameters
  protected final int maxS = 4;
  protected int minF = 4;
  protected int maxF = 6;
  protected static boolean[] NORMALIZATION = new boolean[]{true, false};

  // chi-squared test
  public static double chi = 2;

  // default liblinear parameters
  public static double bias = 1;
  public static double p = 0.1;
  public static int iterations = 5000;
  public static double c = 1;
  public static SolverType solverType = SolverType.L2R_LR_DUAL;

  private double trainAcc = -1;

  public static int MIN_WINDOW_LENGTH = 2;
  public static int MAX_WINDOW_LENGTH = 350;

  // ten-fold cross validation
  private int folds = 10;
  
  private String trainCVPath="";
  private boolean trainCV=false;
  private int seed=0;
  boolean setSeed=false;

  @Override
  public void writeCVTrainToFile(String outputPathAndName) {
    trainCVPath=outputPathAndName;
    trainCV=true;
  }
  
  @Override
  public void setFindTrainAccuracyEstimate(boolean setCV){
    trainCV=setCV;
  }

  @Override
  public ClassifierResults getTrainResults() {     
    return trainResults;
  }
  
  public void setSeed(int s){
      seed = s;
      setSeed = true;
  }

  public static class WEASELModel {

    public WEASELModel(){}

    public WEASELModel(
        boolean normed,
        int features,
        WEASELTransform model,
        de.bwaldvogel.liblinear.Model linearModel
    ) {
      this.normed = normed;
      this.features = features;
      this.weasel = model;
      this.linearModel = linearModel;
    }
    public boolean normed;

    // the best number of Fourier values to be used
    public int features;

    // the trained WEASEL transformation
    public WEASELTransform weasel;

    // the trained liblinear classifier
    public de.bwaldvogel.liblinear.Model linearModel;
  }

  /**
   *
   */
  public WEASEL() {
    super();
  }
  
  public WEASEL(int s) {
    super();
    seed = s;
    setSeed = true;
  }

  @Override
  public String getParameters() {
    StringBuilder sb = new StringBuilder();
    sb.append(super.getParameters());
    sb.append(",maxF,").append(maxF).append(",minF,").append(minF);
    return sb.toString();
  }

  protected int getMax(Instances samples, int maxWindowSize) {
    int max = 0;

    for (Instance inst : samples) {
      max = Math.max(instanceLength(inst), max);
    }
    return Math.min(maxWindowSize,max);
  }

  public int[] getWindowLengths(final Instances samples, boolean norm) {
    int min = norm && MIN_WINDOW_LENGTH<=2? Math.max(3,MIN_WINDOW_LENGTH) : MIN_WINDOW_LENGTH;
    int max = getMax(samples, MAX_WINDOW_LENGTH);

    int[] wLengths = new int[max - min + 1];
    int a = 0;
    for (int w = min; w <= max; w+=1, a++) {
      wLengths[a] = w;
    }
    return Arrays.copyOfRange(wLengths, 0, a);
  }

  protected static double[] getLabels(final WEASELTransform.BagOfBigrams[] bagOfPatternsTestSamples) {
    double[] labels = new double[bagOfPatternsTestSamples.length];
    for (int i = 0; i < bagOfPatternsTestSamples.length; i++) {
      labels[i] = Double.valueOf(bagOfPatternsTestSamples[i].label);
    }
    return labels;
  }

  protected static Problem initLibLinearProblem(
      final WEASELTransform.BagOfBigrams[] bob,
      final WEASELTransform.Dictionary dict,
      final double bias) {
    Linear.resetRandom();

    Problem problem = new Problem();
    problem.bias = bias;
    problem.n = dict.size() + 1;
    problem.y = getLabels(bob);

    final FeatureNode[][] features = initLibLinear(bob, problem.n);

    problem.l = features.length;
    problem.x = features;
    return problem;
  }

  protected static FeatureNode[][] initLibLinear(final WEASELTransform.BagOfBigrams[] bob, int max_feature) {
    FeatureNode[][] featuresTrain = new FeatureNode[bob.length][];
    for (int j = 0; j < bob.length; j++) {
      WEASELTransform.BagOfBigrams bop = bob[j];
      ArrayList<FeatureNode> features = new ArrayList<>(bop.bob.size());
      for (IntIntCursor word : bop.bob) {
        if (word.value > 0 && word.key <= max_feature) {
          features.add(new FeatureNode(word.key, (word.value)));
        }
      }
      FeatureNode[] featuresArray = features.toArray(new FeatureNode[]{});
      Arrays.parallelSort(featuresArray, new Comparator<FeatureNode>() {
        public int compare(FeatureNode o1, FeatureNode o2) {
          return Integer.compare(o1.index, o2.index);
        }
      });
      featuresTrain[j] = featuresArray;
    }
    return featuresTrain;
  }

  private static void swap(int[] array, int idxA, int idxB) {
    int temp = array[idxA];
    array[idxA] = array[idxB];
    array[idxB] = temp;
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

  @Override
  public void buildClassifier(final Instances samples) throws Exception {
    long t1=System.currentTimeMillis();
    
    if(trainCV){
        int numFolds=setNumberOfFolds(samples);
        CrossValidator cv = new CrossValidator();
        if (setSeed)
            cv.setSeed(seed);
        cv.setNumFolds(numFolds);

        WEASEL weasel=new WEASEL();
        trainResults=cv.crossValidateWithStats(weasel,samples);
    }
    
    if (samples.classIndex() != samples.numAttributes()-1)
      throw new Exception("WEASEL_BuildClassifier: Class attribute not set as last attribute in dataset");

    try {
      int maxCorrect = -1;
      int bestF = -1;
      boolean bestNorm = false;

      optimize:
      for (final boolean mean : NORMALIZATION) {
        int[] windowLengths = getWindowLengths(samples, mean);
        WEASELTransform model = new WEASELTransform(maxF, maxS, windowLengths, mean);
        int[][][] words = model.createWords(samples);

        for (int f = minF; f <= maxF; f += 2) {
          model.dict.reset();
          WEASELTransform.BagOfBigrams[] bop = model.createBagOfPatterns(words, samples, f);
          model.filterChiSquared(bop, chi);

          // train liblinear
          final Problem problem = initLibLinearProblem(bop, model.dict, bias);
          int correct = trainLibLinear(problem, solverType, c, iterations, p, folds);

          if (correct > maxCorrect) {
            //System.out.println(correct + "\t" + f);
            maxCorrect = correct;
            bestF = f;
            bestNorm = mean;
          }
          if (correct == samples.numInstances()) {
            break optimize;
          }
        }
      }

      // obtain the final matrix
      int[] windowLengths = getWindowLengths(samples, bestNorm);
      WEASELTransform model = new WEASELTransform(maxF, maxS, windowLengths, bestNorm);

      int[][][] words = model.createWords(samples);
      WEASELTransform.BagOfBigrams[] bob = model.createBagOfPatterns(words, samples, bestF);
      model.filterChiSquared(bob, chi);

      // train liblinear
      Problem problem = initLibLinearProblem(bob, model.dict, bias);
      de.bwaldvogel.liblinear.Model linearModel = Linear.train(problem, new Parameter(solverType, c, iterations, p));

      this.classifier = new WEASELModel(
          bestNorm,
          bestF,
          model,
          linearModel
      );

    } catch (Exception e) {
      e.printStackTrace();
    }
    
    long t2=System.currentTimeMillis();
    trainResults.buildTime=t2-t1;
    
    if(trainCVPath!=""){
        OutFile of=new OutFile(trainCVPath);
        of.writeLine(samples.relationName()+",TSF,train");
        of.writeLine(getParameters());
        of.writeLine(trainResults.acc+"");
        double[] trueClassVals,predClassVals;
        trueClassVals=trainResults.getTrueClassVals();
        predClassVals=trainResults.getPredClassVals();
        for(int i=0;i<samples.numInstances();i++){
            //Basic sanity check
            if(samples.instance(i).classValue()!=trueClassVals[i]){
                throw new Exception("ERROR in TSF cross validation, class mismatch!");
            }
            of.writeString((int)trueClassVals[i]+","+(int)predClassVals[i]+",");
            for(double d:trainResults.getDistributionForInstance(i))
                of.writeString(","+d);
            of.writeString("\n");
        }
    }
  }


  @Override
  public double classifyInstance(Instance instance) throws Exception {
    final int[][] wordsTest = classifier.weasel.createWords(instance);
    WEASELTransform.BagOfBigrams[] bagTest = new WEASELTransform.BagOfBigrams[]{classifier.weasel.createBagOfPatterns(wordsTest, instance, classifier.features)};

    // chi square changes key mappings => remap
    classifier.weasel.dict.remap(bagTest);

    FeatureNode[][] features = initLibLinear(bagTest, classifier.linearModel.getNrFeature());
    return Linear.predict(classifier.linearModel, features[0]);
  }

  @Override
  public double[] distributionForInstance(Instance instance) throws Exception {
    double[] classHist = new double[instance.numClasses()];

    final int[][] wordsTest = classifier.weasel.createWords(instance);
    WEASELTransform.BagOfBigrams[] bagTest = new WEASELTransform.BagOfBigrams[]{classifier.weasel.createBagOfPatterns(wordsTest, instance, classifier.features)};

    // chi square changes key mappings => remap
    classifier.weasel.dict.remap(bagTest);

    FeatureNode[][] features = initLibLinear(bagTest, classifier.linearModel.getNrFeature());

    double[] probabilities = new double[classifier.linearModel.getNrClass()];

    Linear.predictProbability(classifier.linearModel, features[0], probabilities);

    // TODO do we have to remap classes to indices???
    for (int i = 0; i < classifier.linearModel.getLabels().length; i++) {
      classHist[classifier.linearModel.getLabels()[i]] = probabilities[i];
    }
    return classHist;
  }


  @Override
  public double getEnsembleCvAcc() {
    return 0;
  }

  @Override
  public double[] getEnsembleCvPreds() {
    return new double[0];
  }

  @Override
  public Capabilities getCapabilities() {
    throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
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


  protected static int instanceLength(Instance inst) {
    int length = inst.numAttributes();
    if (inst.classIndex() >= 0)
      --length;

    return length;
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


  /**
   * WEASEL classifier to be used with known parameters, for boss with parameter search, use BOSSEnsemble.
   *
   * Current implementation of BitWord as of 07/11/2016 only supports alphabetsize of 4, which is the expected value
   * as defined in the paper
   *
   * Params:
   *
   * @author Patrick Schaefer
   */
  public static class WEASELTransform {

    public int alphabetSize;
    public int maxF;

    public int[] windowLengths;
    public boolean normMean;
    public SFASupervised[] signature;
    public Dictionary dict;

    /**
     * The WEASEL-model: a histogram of SFA word and bi-gram frequencies
     */
    public static class BagOfBigrams {
      public IntIntHashMap bob;
      public Double label;

      public BagOfBigrams(int size, Double label) {
        this.bob = new IntIntHashMap(size);
        this.label = label;
      }
    }

    /**
     * A dictionary that maps each SFA word to an integer.
     * <p>
     * Condenses the SFA word space.
     */
    public static class Dictionary {
      LongIntHashMap dict;
      IntIntHashMap dictChi;

      public Dictionary() {
        this.dict = new LongIntHashMap();
        this.dictChi = new IntIntHashMap();
      }

      public void reset() {
        this.dict = new LongIntHashMap();
        this.dictChi = new IntIntHashMap();
      }

      public int getWord(long word) {
        int index = 0;
        if ((index = this.dict.indexOf(word)) > -1) {
          return this.dict.indexGet(index);
        } else {
          int newWord = this.dict.size() + 1;
          this.dict.put(word, newWord);
          return newWord;
        }
      }

      public int getWordChi(int word) {
        int index = 0;
        if ((index = this.dictChi.indexOf(word)) > -1) {
          return this.dictChi.indexGet(index);
        } else {
          int newWord = this.dictChi.size() + 1;
          this.dictChi.put(word, newWord);
          return newWord;
        }
      }

      public int size() {
        if (!this.dictChi.isEmpty()) {
          return this.dictChi.size();
        } else {
          return this.dict.size();
        }
      }

      public void remap(final BagOfBigrams[] bagOfPatterns) {
        for (int j = 0; j < bagOfPatterns.length; j++) {
          IntIntHashMap oldMap = bagOfPatterns[j].bob;
          bagOfPatterns[j].bob = new IntIntHashMap();
          for (IntIntCursor word : oldMap) {
            if (word.value > 0) {
              bagOfPatterns[j].bob.put(getWordChi(word.key), word.value);
            }
          }
        }
      }
    }

    public WEASELTransform( int maxF, int maxS,
                   int[] windowLengths, boolean normMean) {
      this.maxF = maxF;
      this.alphabetSize = maxS;
      this.windowLengths = windowLengths;
      this.normMean = normMean;
      this.dict = new Dictionary();
      this.signature = new SFASupervised[windowLengths.length];
    }

    /**
     * Create SFA words and bigrams for all samples
     *
     * @param samples
     * @return
     */
    public int[][][] createWords(final Instances samples) {
      // create bag of words for each window queryLength
      final int[][][] words = new int[this.windowLengths.length][samples.numInstances()][];
      for (int w = 0; w < this.windowLengths.length; w++) {
        words[w] = createWords(samples, w);
      };
      return words;
    }

    /**
     * Create SFA words and bigrams for a single sample
     *
     * @param sample
     * @return
     */
    public int[][] createWords(final Instance sample) {
      // create bag of words for each window queryLength
      final int[][] words = new int[this.windowLengths.length][];
      for (int w = 0; w < windowLengths.length; w++) {
        words[w] = createWords(sample, w);
      };
      return words;
    }

    /**
     * Create SFA words and bigrams for all samples
     *
     * @param samples
     * @return
     */
    private int[][] createWords(final Instances samples, final int index) {

      // SFA quantization
      if (this.signature[index] == null) {
        this.signature[index] = new SFASupervised();
        this.signature[index].fitWindowing(
            samples, this.windowLengths[index], this.maxF, this.alphabetSize, this.normMean);
      }

      // create words
      final int[][] words = new int[samples.numInstances()][];
      for (int i = 0; i < samples.numInstances(); i++) {
        words[i] = createWords(samples.get(i), index);
      }

      return words;
    }

    /**
     * Create SFA words and bigrams for a single sample
     *
     * @param sample
     * @return
     */
    private int[] createWords(final Instance sample, final int index) {
      // create words
      if (instanceLength(sample) >= this.windowLengths[index]) {
        return this.signature[index].transformWindowingInt(sample, this.maxF);
      } else {
        return new int[]{};
      }
    }

    /**
     * Implementation based on:
     * https://github.com/scikit-learn/scikit-learn/blob/c957249/sklearn/feature_selection/univariate_selection.py#L170
     */
    public void filterChiSquared(final BagOfBigrams[] bob, double chi_limit) {
      // Chi2 Test
      IntIntHashMap featureCount = new IntIntHashMap(bob[0].bob.size());
      LongFloatHashMap classProb = new LongFloatHashMap(10);
      LongIntHashMap observed = new LongIntHashMap(bob[0].bob.size());

      // count number of samples with this word
      for (BagOfBigrams bagOfPattern : bob) {
        long label = bagOfPattern.label.longValue();
        for (IntIntCursor word : bagOfPattern.bob) {
          if (word.value > 0) {
            featureCount.putOrAdd(word.key, 1, 1);
            long key = label << 32 | word.key;
            observed.putOrAdd(key, 1, 1);
          }
        }
      }

      // samples per class
      for (BagOfBigrams bagOfPattern : bob) {
        long label = bagOfPattern.label.longValue();
        classProb.putOrAdd(label, 1, 1);
      }

      // chi-squared: observed minus expected occurrence
      IntHashSet chiSquare = new IntHashSet(featureCount.size());
      for (LongFloatCursor prob : classProb) {
        prob.value /= bob.length; // (float) frequencies.get(prob.key);

        for (IntIntCursor feature : featureCount) {
          long key = prob.key << 32 | feature.key;
          float expected = prob.value * feature.value;

          float chi = observed.get(key) - expected;
          float newChi = chi * chi / expected;
          if (newChi >= chi_limit
              && !chiSquare.contains(feature.key)) {
            chiSquare.add(feature.key);
          }
        }
      }

      for (int j = 0; j < bob.length; j++) {
        for (IntIntCursor cursor : bob[j].bob) {
          if (!chiSquare.contains(cursor.key)) {
            bob[j].bob.values[cursor.index] = 0;
          }
        }
      }

      // chi-squared reduces keys substantially => remap
      this.dict.remap(bob);
    }

    /**
     * Create words and bi-grams for all window lengths
     */
    public BagOfBigrams[] createBagOfPatterns(
        final int[][][] words,
        final Instances samples,
        final int wordLength) {
      BagOfBigrams[] bagOfPatterns = new BagOfBigrams[samples.numInstances()];

      final byte usedBits = (byte) binlog(this.alphabetSize);
      final long mask = (1L << (usedBits * wordLength)) - 1L;
      int highestBit = binlog(Integer.highestOneBit(MAX_WINDOW_LENGTH))+1;

      // iterate all samples
      // and create a bag of pattern
      for (int j = 0; j < samples.numInstances(); j++) {
        bagOfPatterns[j] = new BagOfBigrams(words[0][j].length * 6, samples.get(j).classValue());

        // create subsequences
        for (int w = 0; w < this.windowLengths.length; w++) {
          for (int offset = 0; offset < words[w][j].length; offset++) {
            int word = this.dict.getWord((words[w][j][offset] & mask) << highestBit | (long) w);
            bagOfPatterns[j].bob.putOrAdd(word, 1, 1);

            // add 2 grams
            if (offset - this.windowLengths[w] >= 0) {
              long prevWord = this.dict.getWord((words[w][j][offset - this.windowLengths[w]] & mask) << highestBit | (long) w);
              int newWord = this.dict.getWord((prevWord << 32 | word ) << highestBit);
              bagOfPatterns[j].bob.putOrAdd(newWord, 1, 1);
            }
          }
        }
      }

      return bagOfPatterns;
    }

    /**
     * Create words and bi-grams for all window lengths
     */
    public BagOfBigrams createBagOfPatterns(
        final int[][] words,
        final Instance sample,
        final int wordLength) {
      final byte usedBits = (byte) binlog(this.alphabetSize);
      final long mask = (1L << (usedBits * wordLength)) - 1L;
      int highestBit = binlog(Integer.highestOneBit(MAX_WINDOW_LENGTH))+1;

      BagOfBigrams bagOfPatterns = new BagOfBigrams(words[0].length * 6, sample.classValue());

      // create subsequences
      for (int w = 0; w < this.windowLengths.length; w++) {
        for (int offset = 0; offset < words[w].length; offset++) {
          int word = this.dict.getWord((words[w][offset] & mask) << highestBit | (long) w);
          bagOfPatterns.bob.putOrAdd(word, 1, 1);

          // add 2 grams
          if (offset - this.windowLengths[w] >= 0) {
            long prevWord = this.dict.getWord((words[w][offset - this.windowLengths[w]] & mask) << highestBit | (long) w);
            int newWord = this.dict.getWord((prevWord << 32 | word ) << highestBit);
            bagOfPatterns.bob.putOrAdd(newWord, 1, 1);
          }
        }
      }
      return bagOfPatterns;
    }
  }

  /**
   * SFA using the ANOVA F-statistic to determine the best Fourier coefficients
   * (those that best separate between class labels) as opposed to using the first
   * ones.
   */
  public static class SFASupervised {
    private static final long serialVersionUID = -6435016083374045799L;

    // distribution of Fourier values
    public transient ArrayList<ValueLabel>[] orderLine;

    public int[] bestValues;

    public int alphabetSize = 256;
    public byte neededBits = (byte) binlog(this.alphabetSize);
    public int wordLength = 0;
    public boolean initialized = false;

    public int maxWordLength;

    // The Momentary Fourier Transform
    public MFT transformation;

    // use binning / bucketing
    public double[][] bins;

    public SFASupervised() {
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

    class ValueLabel {
      public double value;
      public double label;

      public ValueLabel(double key, Double label) {
        this.value = key;
        this.label = label != null? label : 0;
      }

      @Override
      public String toString() {
        return "" + this.value + ":" + this.label;
      }
    }

    /**
     * Extracts sliding windows from the time series and trains SFA based on the sliding windows.
     * At the end of this call, the quantization bins are set.
     *
     * @param timeSeries   A set of samples
     * @param windowLength The queryLength of each sliding window
     * @param wordLength   the SFA word-queryLength
     * @param symbols      the SFA alphabet size
     * @param normMean     if set, the mean is subtracted from each sliding window
     */
    public void fitWindowing(Instances timeSeries, int windowLength, int wordLength, int symbols, boolean normMean) {
      this.transformation = new MFT(windowLength, normMean);

      ArrayList<double[]> sa = new ArrayList<>(timeSeries.numInstances());
      ArrayList<Double> labels = new ArrayList<>(timeSeries.numInstances());

      for (Instance t : timeSeries) {
        for (double[] data : getDisjointSequences(t, windowLength, normMean)) {
          sa.add(data);
          labels.add(t.classValue());
        }
      }

      double[][] allSamples = new double[sa.size()][];
      double[] allLabels = new double[sa.size()];
      for (int i = 0; i < sa.size(); i++) {
        allSamples[i] = sa.get(i);
        allLabels[i] = labels.get(i);
      }

      fitTransform(allSamples, allLabels, wordLength, symbols, normMean);
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
        subsequences[i] = z_norm(subsequenceData, normMean);
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
      mean /= (double) data.length;

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

    protected double calculateInformationGain(
        ObjectIntHashMap<Double> cIn, ObjectIntHashMap<Double> cOut,
        double class_entropy,
        double total_c_in,
        double total) {
      double total_c_out = (total - total_c_in);
      return class_entropy
          - total_c_in / total * entropy(cIn, total_c_in)
          - total_c_out / total * entropy(cOut, total_c_out);
    }

    protected void findBestSplit(
        List<ValueLabel> element,
        int start,
        int end,
        int remainingSymbols,
        List<Integer> splitPoints
    ) {

      double bestGain = -1;
      int bestPos = -1;

      // class entropy
      int total = end - start;
      ObjectIntHashMap<Double> cIn = new ObjectIntHashMap<>();
      ObjectIntHashMap<Double> cOut = new ObjectIntHashMap<>();
      for (int pos = start; pos < end; pos++) {
        cOut.putOrAdd(element.get(pos).label, 1, 1);
      }
      double class_entropy = entropy(cOut, total);

      int i = start;
      Double lastLabel = element.get(i).label;
      i += moveElement(element, cIn, cOut, start);

      for (int split = start + 1; split < end - 1; split++) {
        Double label = element.get(i).label;
        i += moveElement(element, cIn, cOut, split);

        // only inspect changes of the label
        if (!label.equals(lastLabel)) {
          double gain = calculateInformationGain(cIn, cOut, class_entropy, i, total);

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

    protected int moveElement(
        List<ValueLabel> element,
        ObjectIntHashMap<Double> cIn, ObjectIntHashMap<Double> cOut,
        int pos) {
      cIn.putOrAdd(element.get(pos).label, 1, 1);
      cOut.putOrAdd(element.get(pos).label, -1, -1);
      return 1;
    }

    protected int getMaxLength(double[][] samples) {
      int length = 0;
      for (int i = 0; i < samples.length; i++) {
        length = Math.max(samples[i].length, length);
      }
      return length;
    }

    /**
     * Same as fitTransformDouble but returns the SFA words instead of the Fourier
     * transformed time series.
     */
    public short[][] fitTransform(double[][] samples, double[] labels, int wordLength, int symbols, boolean normMean) {
      int length = getMaxLength(samples);
      double[][] transformedSignal = fitTransformDouble(samples, labels, length, symbols, normMean);

      Indices<Double>[] best = calcBestCoefficients(samples, labels, transformedSignal);

      // use best coefficients (the ones with largest f-value)
      this.bestValues = new int[Math.min(best.length, wordLength)];
      this.maxWordLength = 0;
      for (int i = 0; i < this.bestValues.length; i++) {
        this.bestValues[i] = best[i].index;
        this.maxWordLength = Math.max(best[i].index + 1, this.maxWordLength);
      }

      // make sure it is an even number
      this.maxWordLength += this.maxWordLength % 2;

      return transform(samples, labels, transformedSignal);
    }

    public double[][] fitTransformDouble(double[][] samples, double[] labels, int wordLength, int symbols, boolean normMean) {
      if (!this.initialized) {
        init(wordLength, symbols);

        if (this.transformation == null) {
          this.transformation = new MFT(samples[0].length, normMean);
        }
      }

      double[][] transformedSamples = fillOrderline(samples, labels, wordLength);

      divideHistogramInformationGain();
      this.orderLine = null;

      return transformedSamples;
    }

    /**
     * Use information-gain to divide the orderline
     */
    protected void divideHistogramInformationGain() {
      // for each Fourier coefficient: split using maximal information gain
      for (int i = 0; i < this.orderLine.length; i++) {
        List<ValueLabel> element = this.orderLine[i];
        if (!element.isEmpty()) {
          ArrayList<Integer> splitPoints = new ArrayList<>();
          findBestSplit(element, 0, element.size(), this.alphabetSize, splitPoints);

          Collections.sort(splitPoints);

          // apply the split
          for (int j = 0; j < splitPoints.size(); j++) {
            double value = element.get(splitPoints.get(j) + 1).value;
            //          double value = (element.get(splitPoints.get(j)).value + element.get(splitPoints.get(j)+1).value)/2.0;
            this.bins[i][j] = value;
          }
        }
      }
    }

    /**
     * calculate ANOVA F-stat
     * compare : https://github.com/scikit-learn/scikit-learn/blob/c957249/sklearn/feature_selection/univariate_selection.py#L121
     *
     * @param transformedSignal
     * @return
     */
    public static Indices<Double>[] calcBestCoefficients(
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
      List<Indices<Double>> best = new ArrayList<>(f.length);
      for (int i = 0; i < f.length; i++) {
        if (!Double.isNaN(f[i])) {
          best.add(new Indices<>(i, f[i]));
        }
      }
      Collections.sort(best);
      return best.toArray(new Indices[]{});
    }

    /**
     * Fills data in the orderline
     *
     * @param samples A set of samples
     */
    protected double[][] fillOrderline(double[][] samples, double[] labels, int l) {
      double[][] transformedSamples = new double[samples.length][];

      for (int i = 0; i < samples.length; i++) {
        // z-normalization
        z_norm(samples[i], true); // TODO needed here?

        // approximation
        transformedSamples[i] = this.transformation.transform(samples[i], l);

        for (int j = 0; j < transformedSamples[i].length; j++) {
          // round to 2 decimal places to reduce noise
          double value = Math.round(transformedSamples[i][j] * 100.0) / 100.0;
          this.orderLine[j].add(new ValueLabel(value, labels[i]));
        }
      }

      // Sort ascending by value
      for (List<ValueLabel> element : this.orderLine) {
        Collections.sort(element, new Comparator<ValueLabel>() {
          @Override
          public int compare(ValueLabel o1, ValueLabel o2) {
            int comp = Double.compare(o1.value, o2.value);
            if (comp != 0) {
              return comp;
            }
            return Double.compare(o1.label,o2.label);
          }
        });
      }

      return transformedSamples;
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

    static class Indices<E extends Comparable<E>> implements Comparable<Indices<E>> {
      int index;
      E value;

      public Indices(int index, E value) {
        this.index = index;
        this.value = value;
      }

      public int compareTo(Indices<E> o) {
        return o.value.compareTo(this.value); // descending sort!
      }

      @Override
      public String toString() {
        return "(" + this.index + ":" + this.value + ")";
      }
    }

    /**
     * Quantization of a DFT approximation to its SFA word
     *
     * @param approximation the DFT approximation of a time series
     * @return
     */
    public short[] quantization(double[] approximation) {
      short[] signal = new short[Math.min(approximation.length, this.bestValues.length)];

      for (int a = 0; a < signal.length; a++) {
        int i = this.bestValues[a];
        // lookup character:
        short beta = 0;
        for (beta = 0; beta < this.bins[i].length; beta++) {
          if (approximation[i] < this.bins[i][beta]) {
            break;
          }
        }
        signal[a] = beta;
      }

      return signal;
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
    public short[][] transform(double[][] samples, double[] labels, double[][] approximation) {
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

    /**
     * Extracts sliding windows from a time series and transforms it to its SFA
     * word.
     * <p>
     * Returns the SFA words as a single int (compacts the characters into one
     * int).
     */
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

    protected double[] transform(double[] series, int wordLength/*, boolean normalize*/) {
      //taken from FFT.java but
      //return just a double[] size n, { real1, imag1, ... realn/2, imagn/2 }
      //instead of Complex[] size n/2

      //only calculating first wordlength/2 coefficients (output values),
      //and skipping first coefficient if the data is to be normalised
//      int n = windowSize;
//      int outputLength = wordLength/2;
//
//      double[] dft = new double[wordLength];
//      double twoPi = 2.0 * Math.PI / n;
//
//      int startOffset2 = normalize? 0 : startOffset/2;
//      for (int k = startOffset2; k < outputLength; k++) {  // For each output element
//        float sumreal = 0;
//        float sumimag = 0;
//        for (int t = 0; t < Math.min(this.windowSize, series.length); t++) {  // For each input element
//          sumreal +=  series[t]*Math.cos(twoPi * t * k);
//          sumimag +=  series[t]*Math.sin(twoPi * t * k);
//        }
//        dft[(k-startOffset2)*2]   = sumreal;
//        dft[(k-startOffset2)*2+1] = normalize? -sumimag : sumimag;
//      }
//      if (startOffset == 0) {
//        dft[1] = 0;
//      }
//      return dft;

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
          double[] data2 = toArrayNoClass(timeSeries);
          System.arraycopy(data2, 0, dft, 0, Math.min(this.windowSize, data.length));

          this.fft.realForward(dft);
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
      double normalisingFactor = std > 0 ? 1.0 / std : 1.0;
      int sign = 1;
      for (int i = 0; i < copy.length; i++) {
        copy[i] *= sign * normalisingFactor;
        sign *= -1;
      }
      return copy;
    }
  }

  public static void main(String[] args) throws Exception{
    //Minimum working example

    for (String dataset : new String[]{ "Coffee",
        "ECG200",
        "FaceFour",
        "OliveOil",
        "GunPoint",
        "Beef",
        "DiatomSizeReduction",
        "CBF",
        "ECGFiveDays",
        "TwoLeadECG",
        "MoteStrain",
        "ItalyPowerDemand"}) {
      Instances train = ClassifierTools.loadData("/Users/bzcschae/workspace/TSC_TONY_new/TimeSeriesClassification/TSCProblems/" + dataset + "/" + dataset + "_TRAIN.arff");
      Instances test = ClassifierTools.loadData("/Users/bzcschae/workspace/TSC_TONY_new/TimeSeriesClassification/TSCProblems/" + dataset + "/" + dataset + "_TEST.arff");

      Classifier c = new WEASEL();
      c.buildClassifier(train);
      double accuracy = ClassifierTools.accuracy(test, c);
      System.out.println("WEASEL accuracy on " + dataset + " fold 0 = " + accuracy);
    }

  }

}