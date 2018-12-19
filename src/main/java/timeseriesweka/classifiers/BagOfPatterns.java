package timeseriesweka.classifiers;

import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.kNN;
import weka.core.Capabilities;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.TechnicalInformation;
import timeseriesweka.filters.BagOfPatternsFilter;
import timeseriesweka.filters.SAX;

/**
 * Converts instances into Bag Of Patterns form, then gives to a 1NN 
 * 
 * Params: wordLength, alphabetSize, windowLength
 * 
 * @author James
 */
public class BagOfPatterns extends AbstractClassifierWithTrainingData{

    public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
    result.setValue(TechnicalInformation.Field.AUTHOR, "J. Lin and R. Khade and Y. Li");
    result.setValue(TechnicalInformation.Field.TITLE, "Rotation-invariant similarity in time series using bag-of-patterns representation");
    result.setValue(TechnicalInformation.Field.JOURNAL, "Journal of Intelligent Information Systems");
    result.setValue(TechnicalInformation.Field.VOLUME, "39");
    result.setValue(TechnicalInformation.Field.NUMBER,"2");
    result.setValue(TechnicalInformation.Field.PAGES, "287-315");
    result.setValue(TechnicalInformation.Field.YEAR, "2012");
    
    return result;
  }
    
    
    public Instances matrix;
    public kNN knn;
    
    private BagOfPatternsFilter bop;
    private int PAA_intervalsPerWindow;
    private int SAX_alphabetSize;
    private int windowSize;
    
    private FastVector alphabet;
    
    private final boolean useParamSearch; //does user want parameter search to be performed
    
    /**
     * No params given, do parameter search
     */
    public BagOfPatterns() {
        this.PAA_intervalsPerWindow = -1;
        this.SAX_alphabetSize = -1;
        this.windowSize = -1;

        knn = new kNN(); //defaults to 1NN, Euclidean distance

        useParamSearch=true;
    }
    
    /**
     * Params given, use those only
     */
    public BagOfPatterns(int PAA_intervalsPerWindow, int SAX_alphabetSize, int windowSize) {
        this.PAA_intervalsPerWindow = PAA_intervalsPerWindow;
        this.SAX_alphabetSize = SAX_alphabetSize;
        this.windowSize = windowSize;
        
        bop = new BagOfPatternsFilter(PAA_intervalsPerWindow, SAX_alphabetSize, windowSize);       
        knn = new kNN(); //default to 1NN, Euclidean distance
        alphabet = SAX.getAlphabet(SAX_alphabetSize);
        
        useParamSearch=false;
    }
    
    public int getPAA_intervalsPerWindow() {
        return PAA_intervalsPerWindow;
    }

    public int getSAX_alphabetSize() {
        return SAX_alphabetSize;
    }

    public int getWindowSize() {
        return windowSize;
    }
    
    /**
     * @return { numIntervals(word length), alphabetSize, slidingWindowSize } 
     */
    public int[] getParameterArray() {
        return new int[] { PAA_intervalsPerWindow, SAX_alphabetSize, windowSize};
    }
    
    /**
     * Performs cross validation on given data for varying parameter values, returns 
     * parameter set which yielded greatest accuracy
     * 
     * @param data Data to perform cross validation testing on
     * @return { numIntervals, alphabetSize, slidingWindowSize } 
     */
    public static int[] parameterSearch(Instances data) throws Exception {

        //BoP paper window search range suggestion
        int minWinSize = (int)((data.numAttributes()-1) * (15.0/100.0));
        int maxWinSize = (int)((data.numAttributes()-1) * (36.0/100.0));
//        int winInc = 1; //check every size in range
        int winInc = (int)((maxWinSize - minWinSize) / 10.0); //check 10 values within that range
        if (winInc < 1) winInc = 1;

        
        double bestAcc = 0.0;
        
        //default to min of each para range
        //this (so far) matters only to the TSC2018 data set Fungi, where
        //the train set consists of one instance from each class, making it
        //impossible to correctly classify using nearest neighbour
        int bestAlpha = 2, bestWord = 2, bestWindowSize = minWinSize;
        
        for (int alphaSize = 2; alphaSize <= 8; alphaSize++) {
            for (int winSize = minWinSize; winSize <= maxWinSize; winSize+=winInc) {
                for (int wordSize = 2; wordSize <= winSize/2; wordSize*=2) { //lin BoP suggestion
                    BagOfPatterns bop = new BagOfPatterns(wordSize, alphaSize, winSize);
                    double acc = bop.crossValidate(data); //leave-one-out without rebuiding every fold
                    
                    if (acc > bestAcc) {
                        bestAcc = acc;
                        bestAlpha = alphaSize;
                        bestWord = wordSize;
                        bestWindowSize = winSize;
                    }
                }
            }
        }
        
        return new int[] { bestWord, bestAlpha, bestWindowSize};
    }
    
    /**
     * Leave-one-out CV without re-doing identical transformation every fold
     * 
     * @return cv accuracy
     */
    private double crossValidate(Instances data) throws Exception {
        buildClassifier(data);
        
        double correct = 0;
        for (int i = 0; i < data.numInstances(); ++i)
            if (classifyInstance(i) == data.get(i).classValue())
                ++correct;
        
        return correct /  data.numInstances();
    }
    
    @Override
    public void buildClassifier(final Instances data) throws Exception {
        trainResults.buildTime=System.currentTimeMillis();
        if (data.classIndex() != data.numAttributes()-1)
            throw new Exception("LinBoP_BuildClassifier: Class attribute not set as last attribute in dataset");
        
        if (useParamSearch) {
            //find and set params
            int[] params = parameterSearch(data);
            
            this.PAA_intervalsPerWindow = params[0];
            this.SAX_alphabetSize = params[1];
            this.windowSize = params[2];
            
            bop = new BagOfPatternsFilter(PAA_intervalsPerWindow, SAX_alphabetSize, windowSize);
            alphabet = SAX.getAlphabet(SAX_alphabetSize);
        }
        
        //validate
        if (PAA_intervalsPerWindow<0)
            throw new Exception("LinBoP_BuildClassifier: Invalid PAA word size: " + PAA_intervalsPerWindow);
        if (PAA_intervalsPerWindow>windowSize)
            throw new Exception("LinBoP_BuildClassifier: Invalid PAA word size, bigger than sliding window size: "
                    + PAA_intervalsPerWindow + "," + windowSize);
        if (SAX_alphabetSize<0 || SAX_alphabetSize>10)
            throw new Exception("LinBoP_BuildClassifier: Invalid SAX alphabet size (valid=2-10): " + SAX_alphabetSize);
        if (windowSize<0 || windowSize>data.numAttributes()-1)
            throw new Exception("LinBoP_BuildClassifier: Invalid sliding window size: " 
                    + windowSize + " (series length "+ (data.numAttributes()-1) + ")");
        
        //real work
        matrix = bop.process(data); //transform
        knn.buildClassifier(matrix); //give to 1nn
        trainResults.buildTime=System.currentTimeMillis()-trainResults.buildTime;
        
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        //convert to BOP form
        double[] hist = bop.bagToArray(bop.buildBag(instance));
        
        //stuff into Instance
        Instances newInsts = new Instances(matrix, 1); //copy attribute data
        newInsts.add(new SparseInstance(1.0, hist));
        
        return knn.classifyInstance(newInsts.firstInstance());
    }

    /**
     * Used as part of a leave-one-out crossvalidation, to skip having to rebuild 
     * the classifier every time (since n-1 histograms would be identical each time anyway), therefore this classifies 
     * the instance at the index passed while ignoring its own corresponding histogram 
     * 
     * @param test index of instance to classify
     * @return classification
     */
    public double classifyInstance(int test) {
        double bestDist = Double.MAX_VALUE;
        double nn = -1.0;
        
        Instance testInst = matrix.get(test);
        
        for (int i = 0; i < matrix.numInstances(); ++i) {
            if (i == test) //skip 'this' one, leave-one-out
                continue;
            
            double dist = knn.distance(testInst, matrix.get(i)); 
            
            if (dist < bestDist) {
                bestDist = dist;
                nn = matrix.get(i).classValue();
            }
        }
        
        return nn;
    }
    
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        //convert to BOP form
        double[] hist = bop.bagToArray(bop.buildBag(instance));
        
        //stuff into Instance
        Instances newInsts = new Instances(matrix, 1); //copy attribute data
        newInsts.add(new SparseInstance(1.0, hist));
        
        return knn.distributionForInstance(newInsts.firstInstance());
    }
    @Override
    public String getParameters() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.getParameters());
        sb.append(",SAXAlphabetSize,").append(getSAX_alphabetSize()).append(",WindowSize,");
        sb.append(getWindowSize()).append(",PAAIntervals,").append(getPAA_intervalsPerWindow());
        return sb.toString();
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static void main(String[] args){
        basicTest();
    }
    
    public static void basicTest() {
        System.out.println("BOPBasicTest\n");
        try {
            Instances train = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\Car\\Car_TRAIN.arff");
            Instances test = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\Car\\Car_TEST.arff");
//            Instances train = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\BeetleFly\\BeetleFly_TRAIN.arff");
//            Instances test = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\BeetleFly\\BeetleFly_TEST.arff");

            System.out.println(train.relationName());

            BagOfPatterns bop = new BagOfPatterns();
            System.out.println("Training starting");
            long start = System.nanoTime();
            bop.buildClassifier(train);
            double trainTime = (System.nanoTime() - start) / 1000000000.0; //seconds
            System.out.println("Training done (" + trainTime + "s)");

            System.out.print("Params: ");
            for (int p : bop.getParameterArray())
                System.out.print(p + " ");
            System.out.println("");

            System.out.println("\nTesting starting");
            start = System.nanoTime();
            double acc = ClassifierTools.accuracy(test, bop);
            double testTime = (System.nanoTime() - start) / 1000000000.0; //seconds
            System.out.println("Testing done (" + testTime + "s)");

            System.out.println("\nACC: " + acc);
        }
        catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
    }
    
    @Override
    public String toString() { 
        return "BagOfPatterns";
    }
}
