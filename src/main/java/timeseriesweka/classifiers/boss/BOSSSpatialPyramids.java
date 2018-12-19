package timeseriesweka.classifiers.boss;


import fileIO.OutFile;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream; 
import java.io.IOException; 
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import weka.core.TechnicalInformation;

import utilities.generic_storage.ComparablePair;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map.Entry;
import utilities.ClassifierTools;
import utilities.BitWord;
import utilities.TrainAccuracyEstimate;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;
import utilities.ClassifierResults;


/**
 * BOSSSpatialPyramids classifier with parameter search and ensembling, if parameters are known, 
 * use nested 'BOSSSpatialPyramidsIndividual' classifier and directly provide them.
 * 
 * Intended use is with the default constructor, however can force the normalisation 
 * parameter to true/false by passing a boolean, e.g c = new BOSSSpatialPyramids(true)
 * 
 * Params: normalise? (i.e should first fourier coefficient(mean value) be discarded)
 * Alphabetsize fixed to four
 * 
 * @author James Large
 * 
 * Base algorithm information found in BOSS.java
 * Spatial Pyramids based on the algorithm described in getTechnicalInformation()
 */
public class BOSSSpatialPyramids implements Classifier, SaveParameterInfo,TrainAccuracyEstimate {
    
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "Lazebnik, Svetlana and Schmid, Cordelia and Ponce, Jean");
        result.setValue(TechnicalInformation.Field.TITLE, "Beyond bags of features: Spatial pyramid matching for recognizing natural scene categories");
        result.setValue(TechnicalInformation.Field.BOOKTITLE, "Computer Vision and Pattern Recognition, 2006 IEEE Computer Society Conference on");
        result.setValue(TechnicalInformation.Field.VOLUME, "2");
        result.setValue(TechnicalInformation.Field.PAGES, "2169--2178");
        result.setValue(TechnicalInformation.Field.YEAR, "2006");

        return result;
    }
    
    private List<BOSSWindow> classifiers; 

    private final double correctThreshold = 0.92;
//    private int maxEnsembleSize = Integer.MAX_VALUE;
    private int maxEnsembleSize = 100;
    
    
    private final Integer[] wordLengths = { 16, 14, 12, 10, 8 };
    private final Integer[] levels = { 1, 2, 3 };
    private final int alphabetSize = 4;
    
    public enum SerialiseOptions { 
        //dont do any seriealising, run as normal
        NONE, 
        
        //serialise the final boss classifiers which made it into ensemble (does not serialise the entire BOSSEnsembleSP_Redo object)
        //slight runtime cost 
        STORE, 
        
        //serialise the final boss classifiers, and delete from main memory. reload each from ser file when needed in classification. 
        //the most memory used at any one time is therefore ~2 individual boss classifiers during training. 
        //massive runtime cost, order of magnitude 
        STORE_LOAD 
    };
    
    
    private SerialiseOptions serOption = SerialiseOptions.NONE;
    private static String serFileLoc = "BOSSWindowSers\\";
     
    private boolean[] normOptions;
    
    private String trainCVPath;
    private boolean trainCV=false;
    private ClassifierResults res =new ClassifierResults();
    
    /**
     * Providing a particular value for normalisation will force that option, if 
     * whether to normalise should be a parameter to be searched, use default constructor
     * 
     * @param normalise whether or not to normalise by dropping the first Fourier coefficient
     */
    public BOSSSpatialPyramids(boolean normalise) {
        normOptions = new boolean[] { normalise };
    }
    
    /**
     * During buildClassifier(...), will search through normalisation as well as 
     * window size and word length if no particular normalisation option is provided
     */
    public BOSSSpatialPyramids() {
        normOptions = new boolean[] { true, false };
    }  

    public static class BOSSWindow implements Comparable<BOSSWindow>, Serializable { 
        private BOSSSpatialPyramidsIndividual classifier;
        public double accuracy;
        public String filename;
        
        private static final long serialVersionUID = 2L;

        public BOSSWindow(String filename) {
            this.filename = filename;
        }
        
        public BOSSWindow(BOSSSpatialPyramidsIndividual classifer, double accuracy, String dataset) {
            this.classifier = classifer;
            this.accuracy = accuracy;
            buildFileName(dataset);
        }

        public double classifyInstance(Instance inst) throws Exception { 
            return classifier.classifyInstance(inst); 
        }
        
        public double classifyInstance(int test) throws Exception { 
            return classifier.classifyInstance(test); 
        }
        
        private void buildFileName(String dataset) {
            filename = serFileLoc + dataset + "_" + classifier.windowSize + "_" + classifier.wordLength + "_" + classifier.alphabetSize + "_" + classifier.norm + ".ser";
        }
        
        public boolean storeAndClearClassifier() {
            try {
                ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename));
                out.writeObject(this);
                out.close();   
                clearClassifier();
                return true;
            }catch(IOException e) {
                System.out.print("Error serialiszing to " + filename);
                e.printStackTrace();
                return false;
            }
        }
        
        public boolean store() {
            try {
                ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename));
                out.writeObject(this);
                out.close();         
                return true;
            }catch(IOException e) {
                System.out.print("Error serialiszing to " + filename);
                e.printStackTrace();
                return false;
            }
        }
        
        public void clearClassifier() {
            classifier = null;
        }
        
        public boolean load() {
            BOSSWindow bw = null;
            try {
                ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename));
                bw = (BOSSWindow) in.readObject();
                in.close();
                this.accuracy = bw.accuracy;
                this.classifier = bw.classifier;
                return true;
            }catch(IOException i) {
                System.out.print("Error deserialiszing from " + filename);
                i.printStackTrace();
                return false;
            }catch(ClassNotFoundException c) {
                System.out.println("BOSSWindow class not found");
                c.printStackTrace();
                return false;
            }
        }
        
        public boolean deleteSerFile() {
            try {
                File f = new File(filename);
                return f.delete();
            } catch(SecurityException s) {
                System.out.println("Unable to delete, access denied: " + filename);
                s.printStackTrace();
                return false;
            }
        }
        
        /**
         * @return { numIntervals(word length), alphabetSize, slidingWindowSize } 
         */
        public int[] getParameters() { return classifier.getParameters();  }
        public int getWindowSize()   { return classifier.getWindowSize();  }
        public int getWordLength()   { return classifier.getWordLength();  }
        public int getAlphabetSize() { return classifier.getAlphabetSize(); }
        public boolean isNorm()      { return classifier.isNorm(); }
        public double getLevelWeighting() { return classifier.getLevelWeighting(); }
        public int getLevels() { return classifier.getLevels(); }
        
        @Override
        public int compareTo(BOSSWindow other) {
            if (this.accuracy > other.accuracy) 
                return 1;
            if (this.accuracy == other.accuracy) 
                return 0;
            return -1;
        }
    }
    
    @Override
    public void writeCVTrainToFile(String train) {
        trainCVPath=train;
        trainCV=true;
    }
    @Override
    public void setFindTrainAccuracyEstimate(boolean setCV){
        trainCV=setCV;
    }
    
    @Override
    public boolean findsTrainAccuracyEstimate(){ return trainCV;}
    
    @Override
    public ClassifierResults getTrainResults(){
//Temporary : copy stuff into res.acc here
//Not implemented?        res.acc=ensembleCvAcc;
//TO DO: Write the other stats        
        return res;
    }        


    @Override
    public String getParameters() {
        StringBuilder sb = new StringBuilder();
        
        BOSSWindow first = classifiers.get(0);
        sb.append("windowSize=").append(first.getWindowSize()).append("/wordLength=").append(first.getWordLength());
        sb.append("/alphabetSize=").append(first.getAlphabetSize()).append("/norm=").append(first.isNorm());
            
        for (int i = 1; i < classifiers.size(); ++i) {
            BOSSWindow boss = classifiers.get(i);
            sb.append(",windowSize=").append(boss.getWindowSize()).append("/wordLength=").append(boss.getWordLength());
            sb.append("/alphabetSize=").append(boss.getAlphabetSize()).append("/norm=").append(boss.isNorm());
        }
        
        return sb.toString();
    }
    
    @Override
    public int setNumberOfFolds(Instances data){
        return data.numInstances();
    }
    
     /**
     * @return { numIntervals(word length), alphabetSize, slidingWindowSize } for each BOSSWindow in this *built* classifier
     */
    public int[][] getParametersValues() {
        int[][] params = new int[classifiers.size()][];
        int i = 0;
        for (BOSSWindow boss : classifiers) 
            params[i++] = boss.getParameters();
         
        return params;
    }
    
    public void setSerOption(SerialiseOptions option) { 
        serOption = option;
    }
    
    public void setSerFileLoc(String path) {
        serFileLoc = path;
    }
    
    @Override
    public void buildClassifier(final Instances data) throws Exception {
        if (data.classIndex() != data.numAttributes()-1)
            throw new Exception("BOSSEnsembleSP_BuildClassifier: Class attribute not set as last attribute in dataset");
 
        if (serOption == SerialiseOptions.STORE || serOption == SerialiseOptions.STORE_LOAD) {
            DateFormat dateFormat = new SimpleDateFormat("yyyyMMddHHmmss");
            Date date = new Date();
            serFileLoc += data.relationName() + "_" + dateFormat.format(date) + "\\";
            File f = new File(serFileLoc);
            if (!f.isDirectory())
                f.mkdirs();
        }
        
        classifiers = new LinkedList<BOSSWindow>();
        
        
        int numSeries = data.numInstances();
        
        int seriesLength = data.numAttributes()-1; //minus class attribute
        int minWindow = 10;
        int maxWindow = seriesLength; 

        //int winInc = 1; //check every window size in range
        
//        //whats the max number of window sizes that should be searched through
        //double maxWindowSearches = Math.min(200, Math.sqrt(seriesLength)); 
        double maxWindowSearches = seriesLength/4.0;
        int winInc = (int)((maxWindow - minWindow) / maxWindowSearches); 
        if (winInc < 1) winInc = 1;
        
        
        //keep track of current max window size accuracy, constantly check for correctthreshold to discard to save space
        double maxAcc = -1.0;
        //the acc of the worst member to make it into the final ensemble as it stands
        double minMaxAcc = -1.0; 
        
        for (boolean normalise : normOptions) {
            for (int winSize = minWindow; winSize <= maxWindow; winSize += winInc) {          
                BOSSSpatialPyramidsIndividual boss = new BOSSSpatialPyramidsIndividual(wordLengths[0], alphabetSize, winSize, normalise, levels[0]); //1 level, find best 'normal' boss classifier
                boss.buildClassifier(data); //initial setup for this windowsize, with max word length     

                BOSSSpatialPyramidsIndividual bestClassifierForWinSize = null; 
                double bestAccForWinSize = -1.0;

                //find best word length for this window size
                for (Integer wordLen : wordLengths) {            
                    boss = boss.buildShortenedSPBags(wordLen); //in first iteration, same lengths (wordLengths[0]), will do nothing

                    int correct = 0; 
                    for (int i = 0; i < numSeries; ++i) {
                        double c = boss.classifyInstance(i); //classify series i, while ignoring its corresponding histogram i
                        if (c == data.get(i).classValue())
                            ++correct;
                    }

                    double acc = (double)correct/(double)numSeries;     
                    if (acc >= bestAccForWinSize) {
                        bestAccForWinSize = acc;
                        bestClassifierForWinSize = boss;
                    }
                }
                
                //best 'normal' boss classifier found, now find the best number of levels
                //effectively determining whether the feature this member/classifier specialises in is 
                //local or global
                int bestLevels = bestClassifierForWinSize.getLevels();
                for (int l = 1; l < levels.length; ++l) { //skip first, already found acc for it before 
                    bestClassifierForWinSize.changeNumLevels(levels[l]);
                    
                    int correct = 0; 
                    for (int i = 0; i < numSeries; ++i) {
                        double c = bestClassifierForWinSize.classifyInstance(i); //classify series i, while ignoring its corresponding histogram i
                        if (c == data.get(i).classValue())
                            ++correct;
                    }

                    double acc = (double)correct/(double)numSeries;     
                    if (acc > bestAccForWinSize) { //only store if >, not >= (favours lower levels = less space)
                        bestAccForWinSize = acc;
                        bestLevels = levels[l];
                    }
                }

                if (makesItIntoEnsemble(bestAccForWinSize, maxAcc, minMaxAcc, classifiers.size())) {
                    bestClassifierForWinSize.changeNumLevels(bestLevels);
                    
                    BOSSWindow bw = new BOSSWindow(bestClassifierForWinSize, bestAccForWinSize, data.relationName());
                    bw.classifier.clean();
                    
                    if (serOption == SerialiseOptions.STORE)
                        bw.store();
                    else if (serOption == SerialiseOptions.STORE_LOAD)
                        bw.storeAndClearClassifier();
                        
                    classifiers.add(bw);
                    
                    if (bestAccForWinSize > maxAcc) {
                        maxAcc = bestAccForWinSize;       
                        //get rid of any extras that dont fall within the final max threshold
                        Iterator<BOSSWindow> it = classifiers.iterator();
                        while (it.hasNext()) {
                            BOSSWindow b = it.next();
                            if (b.accuracy < maxAcc * correctThreshold) {
                                if (serOption == SerialiseOptions.STORE || serOption == SerialiseOptions.STORE_LOAD)
                                    b.deleteSerFile();
                                it.remove();
                            }
                        }
                    }
                    
                    while (classifiers.size() > maxEnsembleSize) {
                        //cull the 'worst of the best' until back under the max size
                        int minAccInd = (int)findMinEnsembleAcc()[0];

                        if (serOption == SerialiseOptions.STORE || serOption == SerialiseOptions.STORE_LOAD)
                            classifiers.get(minAccInd).deleteSerFile();
                        classifiers.remove(minAccInd);
                    }
                    minMaxAcc = findMinEnsembleAcc()[1]; //new 'worst of the best' acc
                }
            }
        }
        
        if (trainCV) {
            int folds=setNumberOfFolds(data);
            OutFile of=new OutFile(trainCVPath);
            of.writeLine(data.relationName()+",BOSSEnsembleSP_Redo,train");
           
            double[][] results = findEnsembleTrainAcc(data);
            of.writeLine(getParameters());
            of.writeLine(results[0][0]+"");
            for(int i=1;i<results[0].length;i++)
                of.writeLine(results[0][i]+","+results[1][i]);
            System.out.println("CV acc ="+results[0][0]);
        }
    }
    
    //[0] = index, [1] = acc
    private double[] findMinEnsembleAcc() {
        double minAcc = Double.MIN_VALUE;
        int minAccInd = 0;
        for (int i = 0; i < classifiers.size(); ++i) {
            double curacc = classifiers.get(i).accuracy;
            if (curacc < minAcc) {
                minAcc = curacc;
                minAccInd = i;
            }
        }
        
        return new double[] { minAccInd, minAcc };
    }
    
    private boolean makesItIntoEnsemble(double acc, double maxAcc, double minMaxAcc, int curEnsembleSize) {
        if (acc >= maxAcc * correctThreshold) {
            if (curEnsembleSize >= maxEnsembleSize)
                return acc > minMaxAcc;
            else 
                return true;
        }
        
        return false;
    }

    private double[][] findEnsembleTrainAcc(Instances data) throws Exception {
        
        double[][] results = new double[2][data.numInstances() + 1];
        
        double correct = 0; 
        for (int i = 0; i < data.numInstances(); ++i) {
            double c = classifyInstance(i, data.numClasses()); //classify series i, while ignoring its corresponding histogram i
            if (c == data.get(i).classValue())
                ++correct;
            
            results[0][i+1] = data.get(i).classValue();
            results[1][i+1] = c;
        }
        
        results[0][0] = correct / data.numInstances();
        //TODO fill results[1][0]
        
        return results;
    }
    
    /**
     * Classify the train instance at index 'test', whilst ignoring the corresponding bags 
     * in each of the members of the ensemble, for use in CV of BOSSEnsembleSP_Redo
     */
    public double classifyInstance(int test, int numclasses) throws Exception {
        double[] dist = distributionForInstance(test, numclasses);
        
        double maxFreq=dist[0], maxClass=0;
        for (int i = 1; i < dist.length; ++i) 
            if (dist[i] > maxFreq) {
                maxFreq = dist[i];
                maxClass = i;
            }
        
        return maxClass;
    }

    public double[] distributionForInstance(int test, int numclasses) throws Exception {
        double[] classHist = new double[numclasses];
        
        //get votes from all windows 
        double sum = 0;
        for (BOSSWindow classifier : classifiers) {
            if (serOption == SerialiseOptions.STORE_LOAD)
                classifier.load();
            double classification = classifier.classifyInstance(test);
            if (serOption == SerialiseOptions.STORE_LOAD)
                classifier.clearClassifier();
            classHist[(int)classification]++;
            sum++;
        }
        
        if (sum != 0)
            for (int i = 0; i < classHist.length; ++i)
                classHist[i] /= sum;
        
        return classHist;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] dist = distributionForInstance(instance);
        
        double maxFreq=dist[0], maxClass=0;
        for (int i = 1; i < dist.length; ++i) 
            if (dist[i] > maxFreq) {
                maxFreq = dist[i];
                maxClass = i;
            }
        
        return maxClass;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] classHist = new double[instance.numClasses()];
        double sum = 0;
        
        //get votes from all windows 
        for (BOSSWindow classifier : classifiers) {
            if (serOption == SerialiseOptions.STORE_LOAD)
                classifier.load();
            double classification = classifier.classifyInstance(instance);
            if (serOption == SerialiseOptions.STORE_LOAD)
                classifier.clearClassifier();
            
            classHist[(int)classification]++;
            sum++;
        }
        
        if (sum != 0)
            for (int i = 0; i < classHist.length; ++i)
                classHist[i] /= sum;
        
        return classHist;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

     public static void main(String[] args) throws Exception{
        //Minimum working example
        String dataset = "ItalyPowerDemand";
        Instances train = ClassifierTools.loadData("C:\\TSC Problems\\"+dataset+"\\"+dataset+"_TRAIN.arff");
        Instances test = ClassifierTools.loadData("C:\\TSC Problems\\"+dataset+"\\"+dataset+"_TEST.arff");
        
        Classifier c = new BOSSSpatialPyramids();
        c.buildClassifier(train);
        double accuracy = ClassifierTools.accuracy(test, c);
        
        System.out.println("BOSSEnsembleSP accuracy on " + dataset + " fold 0 = " + accuracy);
        
        //Other examples/tests
//        detailedFold0Test(dataset);
//        resampleTest(dataset, 25);
    }
    
    public static void detailedFold0Test(String dset) {
        System.out.println("BOSSEnsembleSPDetailedTest\n");
        try {
            Instances train = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TRAIN.arff");
            Instances test = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TEST.arff");
            System.out.println(train.relationName());
            
            BOSSSpatialPyramids boss = new BOSSSpatialPyramids();
            
            //TRAINING
            System.out.println("Training starting");
            long start = System.nanoTime();
            boss.buildClassifier(train);
            double trainTime = (System.nanoTime() - start) / 1000000000.0; //seconds
            System.out.println("Training done (" + trainTime + "s)");
            
            //RESULTS OF TRAINING
            System.out.println("Ensemble Size: " + boss.classifiers.size());
            System.out.println("Param sets: ");
            int[][] params = boss.getParametersValues();
            for (int i = 0; i < params.length; ++i)
                System.out.println(i + ": " + params[i][0] + " " + params[i][1] + " " + params[i][2] + " " + boss.classifiers.get(i).isNorm() + " " + boss.classifiers.get(i).getLevels() + " " + boss.classifiers.get(i).accuracy);
            
            //TESTING
            System.out.println("\nTesting starting");
            start = System.nanoTime();
            double acc = ClassifierTools.accuracy(test, boss);
            double testTime = (System.nanoTime() - start) / 1000000000.0; //seconds
            System.out.println("Testing done (" + testTime + "s)");
            
            System.out.println("\nACC: " + acc);
        }
        catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
    }
        
    public static void resampleTest(String dset, int resamples) throws Exception {
        Instances train = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TRAIN.arff");
        Instances test = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TEST.arff");
         
        Classifier c = new BOSSSpatialPyramids();
         
        //c.setCVPath("C:\\tempproject\\BOSSEnsembleCVtest.csv");
         
        double [] accs = new double[resamples];
         
        for(int i=0;i<resamples;i++){
            Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, i);
            
            c.buildClassifier(data[0]);
            accs[i]= ClassifierTools.accuracy(data[1], c);
            
            if (i==0)
                System.out.print(accs[i]);
            else 
                System.out.print("," + accs[i]);
        }
         
        double mean = 0;
        for(int i=0;i<resamples;i++)
            mean += accs[i];
        mean/=resamples;
         
        System.out.println("\n\nBOSSEnsembleSP mean acc over " + resamples + " resamples: " + mean);
    }
    


    /**
     * BOSSSpatialPyramidsIndividual classifier to be used with known parameters, 
     * for boss with parameter search, use BOSSSpatialPyramids.
     * 
     * Params: wordLength, alphabetSize, windowLength, normalise?
     * 
     * @author James Large. 
     */
    public static class BOSSSpatialPyramidsIndividual implements Classifier, Serializable {

        protected BitWord [][] SFAwords; //all sfa words found in original buildClassifier(), no numerosity reduction/shortening applied
        public ArrayList<SPBag> bags; //histograms of words of the current wordlength with numerosity reduction applied (if selected)
        protected double[/*letterindex*/][/*breakpointsforletter*/] breakpoints;

        protected double inverseSqrtWindowSize;
        protected int windowSize;
        protected int wordLength;
        protected int alphabetSize;
        protected boolean norm;

        protected int levels = 0;
        protected double levelWeighting = 0.5;
        protected int seriesLength;

        protected boolean numerosityReduction = true; 

        protected static final long serialVersionUID = 1L;

        public BOSSSpatialPyramidsIndividual(int wordLength, int alphabetSize, int windowSize, boolean normalise, int levels) {
            this.wordLength = wordLength;
            this.alphabetSize = alphabetSize;
            this.windowSize = windowSize;
            this.inverseSqrtWindowSize = 1.0 / Math.sqrt(windowSize);
            this.norm = normalise;

            this.levels = levels;
        }

        /**
         * Used when shortening histograms, copies 'meta' data over, but with shorter 
         * word length, actual shortening happens separately
         */
        public BOSSSpatialPyramidsIndividual(BOSSSpatialPyramidsIndividual boss, int wordLength) {
            this.wordLength = wordLength;

            this.windowSize = boss.windowSize;
            this.inverseSqrtWindowSize = boss.inverseSqrtWindowSize;
            this.alphabetSize = boss.alphabetSize;
            this.norm = boss.norm;
            this.numerosityReduction = boss.numerosityReduction; 
            //this.alphabet = boss.alphabet;

            this.SFAwords = boss.SFAwords;
            this.breakpoints = boss.breakpoints;

            this.levelWeighting = boss.levelWeighting;
            this.levels = boss.levels;
            this.seriesLength = boss.seriesLength;

            bags = new ArrayList<>(boss.bags.size());
        }

        //map of <word, level> => count
        public static class SPBag extends HashMap<ComparablePair<BitWord, Integer>, Double> {
            double classVal;

            public SPBag() {
                super();
            }

            public SPBag(int classValue) {
                super();
                classVal = classValue;
            }

            public double getClassVal() { return classVal; }
            public void setClassVal(double classVal) { this.classVal = classVal; }       
        }

        public int getWindowSize() { return windowSize; }
        public int getWordLength() { return wordLength; }
        public int getAlphabetSize() { return alphabetSize; }
        public boolean isNorm()     { return norm; }
        public int getLevels()   { return levels; }
        public double getLevelWeighting() { return levelWeighting; }

        /**
         * @return { numIntervals(word length), alphabetSize, slidingWindowSize, normalise? } 
         */
        public int[] getParameters() {
            return new int[] { wordLength, alphabetSize, windowSize };
        }

        public void clean() {
            SFAwords = null;
        }

        protected double[][] slidingWindow(double[] data) {
            int numWindows = data.length-windowSize+1;
            double[][] subSequences = new double[numWindows][windowSize];

            for (int windowStart = 0; windowStart < numWindows; ++windowStart) { 
                //copy the elements windowStart to windowStart+windowSize from data into 
                //the subsequence matrix at row windowStart
                System.arraycopy(data,windowStart,subSequences[windowStart],0,windowSize);
            }

            return subSequences;
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

        /**
         * Performs DFT but calculates only wordLength/2 coefficients instead of the 
         * full transform, and skips the first coefficient if it is to be normalised
         * 
         * @return double[] size wordLength, { real1, imag1, ... realwl/2, imagwl/2 }
         */
        protected double[] DFT(double[] series) {
            //taken from FFT.java but 
            //return just a double[] size n, { real1, imag1, ... realn/2, imagn/2 }
            //instead of Complex[] size n/2
            
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

            int n=series.length;
            int outputLength = wordLength/2;
            int start = (norm ? 1 : 0);

            //normalize the disjoint windows and sliding windows by dividing them by their standard deviation 
            //all Fourier coefficients are divided by sqrt(windowSize)

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
          for (int i = 0; i < dft.length; i++) {
            dft[i] *= normalisingFactor;
          }
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
            calcIncreamentalMeanStddev(windowSize, series, means, stds);
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
        private void calcIncreamentalMeanStddev(int windowLength, double[] series, double[] means, double[] stds) {
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
            for (Instance inst : data) {
                dfts[sample++] = performDFT(disjointWindows(toArrayNoClass(inst))); //approximation
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

        /**
         * Builds a brand new boss bag from the passed fourier transformed data, rather than from
         * looking up existing transforms from earlier builds. 
         * 
         * to be used e.g to transform new test instances
         */
        protected SPBag createSPBagSingle(double[][] dfts) {
            SPBag bag = new SPBag();
            BitWord lastWord = new BitWord();

            int wInd = 0;
            int trivialMatchCount = 0;

            for (double[] d : dfts) {
                BitWord word = createWord(d);

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

        protected BitWord createWord(double[] dft) {
            BitWord word = new BitWord(wordLength);
            for (int l = 0; l < wordLength; ++l) {//for each letter
                for (int bp = 0; bp < alphabetSize; ++bp) {//run through breakpoints until right one found
                    if (dft[l] <= breakpoints[l][bp]) {
                        word.push(bp); //add corresponding letter to word
                        break;
                    }
                }
            }

            return word;
        }

        /**
         * Assumes class index, if present, is last
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
        public BOSSSpatialPyramidsIndividual buildShortenedSPBags(int newWordLength) throws Exception {
            if (newWordLength == wordLength) //case of first iteration of word length search in ensemble
                return this;
            if (newWordLength > wordLength)
                throw new Exception("Cannot incrementally INCREASE word length, current:"+wordLength+", requested:"+newWordLength);
            if (newWordLength < 2)
                throw new Exception("Invalid wordlength requested, current:"+wordLength+", requested:"+newWordLength);

            BOSSSpatialPyramidsIndividual newBoss = new BOSSSpatialPyramidsIndividual(this, newWordLength);

            //build hists with new word length from SFA words, and copy over the class values of original insts
            for (int i = 0; i < bags.size(); ++i) {
                SPBag newSPBag = createSPBagFromWords(newWordLength, SFAwords[i], true);   
                newSPBag.setClassVal(bags.get(i).getClassVal());
                newBoss.bags.add(newSPBag);
            }

            return newBoss;
        }

        protected SPBag shortenSPBag(int newWordLength, int bagIndex) {
            SPBag newSPBag = new SPBag();

            for (BitWord word : SFAwords[bagIndex]) {
                BitWord shortWord = new BitWord(word);
                shortWord.shortenByFourierCoefficient();

                Double val = newSPBag.get(shortWord);
                if (val == null)
                    val = 0.0;

                newSPBag.put(new ComparablePair<BitWord, Integer>(shortWord, 0), val + 1.0);
            }

            return newSPBag;
        }

        /**
         * Builds a bag from the set of words for a pre-transformed series of a given wordlength.
         * @param wordLengthSearching if true, length of each SFAwords word assumed to be 16, 
         *      and need to shorten it to whatever actual value needed in this particular version of the 
         *      classifier. if false, this is a standalone classifier with pre-defined wordlength (etc),
         *      and therefore sfawords are that particular length already, no need to shorten
         */
        protected SPBag createSPBagFromWords(int thisWordLength, BitWord[] words, boolean wordLengthSearching) {
            SPBag bag = new SPBag();
            BitWord lastWord = new BitWord();

            int wInd = 0;
            int trivialMatchCount = 0; //keeps track of how many words have been the same so far

            for (BitWord w : words) {
                BitWord word = new BitWord(w);
                if (wordLengthSearching)
                    word.shorten(16-thisWordLength); //TODO hack, word.length=16=maxwordlength, wordLength of 'this' BOSSSpatialPyramids instance unreliable, length of SFAwords = maxlength

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
                SPBag bag = createSPBagFromWords(wordLength, SFAwords[inst], true); //rebuild bag
                bag.setClassVal(bags.get(inst).classVal);
                bags.set(inst, bag); //overwrite old
            }
        }

        protected void applyPyramidWeights(SPBag bag) {
            for (Entry<ComparablePair<BitWord, Integer>, Double> ent : bag.entrySet()) {
                //find level that this quadrant is on
                int quadrant = ent.getKey().var2;
                int qEnd = 0; 
                int level = 0; 
                while (qEnd < quadrant) {
                    int numQuadrants = (int)Math.pow(2, ++level);
                    qEnd+=numQuadrants;
                }

                double val = ent.getValue() * (Math.pow(levelWeighting, levels-level-1)); //weighting ^ (levels - level)
                bag.put(ent.getKey(), val); 
            }
        }

        protected void addWordToPyramid(BitWord word, int wInd, SPBag bag) {
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

                ComparablePair<BitWord, Integer> key = new ComparablePair<>(word, quadrant);
                Double val = bag.get(key);

                if (val == null)
                    val = 0.0;
                bag.put(key, ++val);   

                qStart += numQuadrants;
            }
        }

        protected BitWord[] createSFAwords(Instance inst) throws Exception {
            double[][] dfts2 = performMFT(toArrayNoClass(inst)); //approximation     
            BitWord[] words2 = new BitWord[dfts2.length];
            for (int window = 0; window < dfts2.length; ++window) 
                words2[window] = createWord(dfts2[window]);//discretisation

            return words2;
        }

        @Override
        public void buildClassifier(Instances data) throws Exception {
            if (data.classIndex() != data.numAttributes()-1)
                throw new Exception("BOSSSpatialPyramids_BuildClassifier: Class attribute not set as last attribute in dataset");

            seriesLength = data.numAttributes()-1;

            breakpoints = MCB(data); //breakpoints to be used for making sfa words for train AND test data

            SFAwords = new BitWord[data.numInstances()][];
            bags = new ArrayList<>(data.numInstances());

            for (int inst = 0; inst < data.numInstances(); ++inst) {
                SFAwords[inst] = createSFAwords(data.get(inst));

                SPBag bag = createSPBagFromWords(wordLength, SFAwords[inst], false);
                bag.setClassVal(data.get(inst).classValue());
                bags.add(bag);
            }
        }

        /**
         * Computes BOSSSpatialPyramids distance between two bags d(test, train), is NON-SYMETRIC operation, ie d(a,b) != d(b,a)
         * @return distance FROM instA TO instB
         */
        public double BOSSSpatialPyramidsDistance(SPBag instA, SPBag instB) {
            double dist = 0.0;

            //find dist only from values in instA
            for (Entry<ComparablePair<BitWord, Integer>, Double> entry : instA.entrySet()) {
                Double valA = entry.getValue();
                Double valB = instB.get(entry.getKey());
                if (valB == null)
                    valB = 0.0;
                dist += (valA-valB)*(valA-valB);
            }

            return dist;
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
            for (Entry<ComparablePair<BitWord, Integer>, Double> entry : instA.entrySet()) {
                Double valA = entry.getValue();
                Double valB = instB.get(entry.getKey());
                if (valB == null)
                    valB = 0.0;
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

            for (Entry<ComparablePair<BitWord, Integer>, Double> entry : instA.entrySet()) {
                Double valA = entry.getValue();
                Double valB = instB.get(entry.getKey());
                if (valB == null)
                    continue;

                sim += Math.min(valA,valB);
            }

            return sim;
        }

        @Override
        public double classifyInstance(Instance instance) throws Exception {
            SPBag testSPBag = BOSSSpatialPyramidsTransform(instance);

            double bestSimilarity = 0.0;
            double nn = -1.0;

            for (int i = 0; i < bags.size(); ++i) {
                double similarity = histogramIntersection(testSPBag, bags.get(i));

                if (similarity > bestSimilarity) {
                    bestSimilarity = similarity;
                    nn = bags.get(i).getClassVal();
                }
            }

            //if no bags had ANY similarity, just randomly guess 0
            //found that this occurs in <1% of test cases for certain parameter sets 
            //in the ensemble 
            if (nn == -1.0)
                nn = 0.0; 

            return nn;
        }

        /**
         * Used within BOSSSpatialPyramidsEnsemble as part of a leave-one-out crossvalidation, to skip having to rebuild 
         * the classifier every time (since the n histograms would be identical each time anyway), therefore this classifies 
         * the instance at the index passed while ignoring its own corresponding histogram 
         * 
         * @param test index of instance to classify
         * @return classification
         */
        public double classifyInstance(int test) {
            double bestSimilarity = 0.0;
            double nn = -1.0;

            SPBag testSPBag = bags.get(test);

            for (int i = 0; i < bags.size(); ++i) {
                if (i == test) //skip 'this' one, leave-one-out
                    continue;

                double similarity = histogramIntersection(testSPBag, bags.get(i));

                if (similarity > bestSimilarity) {
                    bestSimilarity = similarity;
                    nn = bags.get(i).getClassVal();
                }
            }

            return nn;
        }

        @Override
        public double[] distributionForInstance(Instance instance) throws Exception {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public Capabilities getCapabilities() {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }
        
        public static void detailedFold0Test(String dset) {
            System.out.println("BOSSSpatialPyramidsIndividual DetailedTest\n");
            try {
                Instances train = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TRAIN.arff");
                Instances test = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TEST.arff");
                System.out.println(train.relationName());

                int windowSize = 10;
                int alphabetSize = 4;
                int wordLength = 58;
                int levels = 2;
                boolean norm = true;

                BOSSSpatialPyramidsIndividual boss = new BOSSSpatialPyramidsIndividual(windowSize, alphabetSize, wordLength, norm, levels);
                System.out.println(boss.getWordLength() + " " + boss.getAlphabetSize() + " " + boss.getWindowSize() + " " + boss.isNorm());

                System.out.println("Training starting");
                long start = System.nanoTime();
                boss.buildClassifier(train);
                double trainTime = (System.nanoTime() - start) / 1000000000.0; //seconds
                System.out.println("Training done (" + trainTime + "s)");

                System.out.println("Breakpoints: ");
                for (int i = 0; i < boss.breakpoints.length; i++) {
                    System.out.print("Letter "  + i + ": ");
                    for (int j = 0; j < boss.breakpoints[i].length; j++) {
                        System.out.print(boss.breakpoints[i][j] + " ");
                    }
                    System.out.println("");
                }

                System.out.println("\nTesting starting");
                start = System.nanoTime();
                double acc = ClassifierTools.accuracy(test, boss);
                double testTime = (System.nanoTime() - start) / 1000000000.0; //seconds
                System.out.println("Testing done (" + testTime + "s)");

                System.out.println("\nACC: " + acc);
            }
            catch (Exception e) {
                System.out.println(e);
                e.printStackTrace();
            }
        }
    }

}