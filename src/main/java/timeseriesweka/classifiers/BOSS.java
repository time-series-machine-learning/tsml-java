package timeseriesweka.classifiers;



import fileIO.OutFile;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import timeseriesweka.classifiers.cote.HiveCoteModule;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
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
import weka.core.TechnicalInformation;

/**
 * BOSS classifier with parameter search and ensembling, if parameters are known, 
 * use the nested class BOSSIndividual and directly provide them.
 * 
 * Intended use is with the default constructor, however can force the normalisation 
 * parameter to true/false by passing a boolean, e.g c = new BOSSEnsemble(true)
 * 
 * Alphabetsize fixed to four
 * 
 * @author James Large
 * 
 * Implementation based on the algorithm described in getTechnicalInformation()
 */
public class BOSS extends AbstractClassifierWithTrainingData implements HiveCoteModule, TrainAccuracyEstimate {
  
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "P. Schafer");
        result.setValue(TechnicalInformation.Field.TITLE, "The BOSS is concerned with time series classification in the presence of noise");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Data Mining and Knowledge Discovery");
        result.setValue(TechnicalInformation.Field.VOLUME, "29");
        result.setValue(TechnicalInformation.Field.NUMBER,"6");
        result.setValue(TechnicalInformation.Field.PAGES, "1505-1530");
        result.setValue(TechnicalInformation.Field.YEAR, "2015");

        return result;
    }
    
    private List<BOSSWindow> classifiers; 
    
    private final double correctThreshold = 0.92;
    private int maxEnsembleSize = Integer.MAX_VALUE;
    
    private final Integer[] wordLengths = { 16, 14, 12, 10, 8 };
    private final int alphabetSize = 4;
    
    private boolean loadFeatureSets = false;
    private int fold = 0;
  
    public enum SerialiseOptions { 
        //dont do any seriealising, run as normal
        NONE, 
        
        //serialise the final boss classifiers which made it into ensemble (does not serialise the entire BOSSEnsemble object)
        //slight runtime cost 
        STORE, 
        
        //serialise the final boss classifiers, and delete from main memory. reload each from ser file when needed in classification. 
        //the most memory used at any one time is therefore ~2 individual boss classifiers during training. 
        //massive runtime cost, order of magnitude 
        STORE_LOAD 
    };
    
    
    private SerialiseOptions serOption = SerialiseOptions.NONE;
    private static String serFileLoc = "BOSSWindowSers\\";
    private static String featureFileLoc = "C:/JamesLPHD/featuresets/BOSSEnsemble/";
     
    private boolean[] normOptions;
    
    private String trainCVPath;
    private boolean trainCV=false;

    private Instances train;
    private double ensembleCvAcc = -1;
    private double[] ensembleCvPreds = null;

    @Override
    public void writeCVTrainToFile(String outputPathAndName){
        trainCVPath=outputPathAndName;
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
//Temporary : copy stuff into trainResults.acc here
        trainResults.acc=ensembleCvAcc;
//TO DO: Write the other stats        
        return trainResults;
    }        
    
    
    /**
     * Providing a particular value for normalisation will force that option, if 
     * whether to normalise should be a parameter to be searched, use default constructor
     * 
     * @param normalise whether or not to normalise by dropping the first Fourier coefficient
     */
    public BOSS(boolean normalise) {
        normOptions = new boolean[] { normalise };
    }
    
    /**
     * During buildClassifier(...), will search through normalisation as well as 
     * window size and word length if no particular normalisation option is provided
     */
    public BOSS() {
        normOptions = new boolean[] { true, false };
    }  

    public static class BOSSWindow implements Comparable<BOSSWindow>, Serializable { 
        private BOSSIndividual classifier;
        public double accuracy;
        public String filename;
        
        private static final long serialVersionUID = 2L;

        public BOSSWindow(String filename) {
            this.filename = filename;
        }
        
        public BOSSWindow(BOSSIndividual classifer, double accuracy, String dataset) {
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
            filename = serFileLoc + dataset + "_" + classifier.windowSize + "_" + 
                    classifier.wordLength + "_" + classifier.alphabetSize + "_" + classifier.norm + ".ser";
        }
        
        public boolean storeAndClearClassifier() {
            try {
                ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename));
                out.writeObject(this);
                out.close();   
                clearClassifier();
                return true;
            }catch(IOException e) {
                System.out.print("Error serialising to " + filename);
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
    public String getParameters() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.getParameters());
        BOSSWindow first = classifiers.get(0);
        sb.append(",windowSize,").append(first.getWindowSize()).append(",wordLength,").append(first.getWordLength());
        sb.append(",alphabetSize,").append(first.getAlphabetSize()).append(",norm,").append(first.isNorm());
            
        for (int i = 1; i < classifiers.size(); ++i) {
            BOSSWindow boss = classifiers.get(i);
            sb.append(",windowSize,").append(boss.getWindowSize()).append(",wordLength,").append(boss.getWordLength());
            sb.append(",alphabetSize,").append(boss.getAlphabetSize()).append(",norm,").append(boss.isNorm());
        }
        
        return sb.toString();
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
    
    public void setFeatureFileLoc(String path) {
        featureFileLoc = path;
    }
    
    public void setMaxEnsembleSize(int max) {
        maxEnsembleSize = max;
    }
    
    public void setLoadFeatures(boolean load, int fold) {
        this.loadFeatureSets = load;
        this.fold = fold;
    }
    
    @Override
    public void buildClassifier(final Instances data) throws Exception {
        trainResults.buildTime=System.currentTimeMillis();
        this.train=data;
        
        if (data.classIndex() != data.numAttributes()-1)
            throw new Exception("BOSSEnsemble_BuildClassifier: Class attribute not set as last attribute in dataset");
 
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
        
        //whats the max number of window sizes that should be searched through
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
                BOSSIndividual boss = null;
                
                //for feature saving/loading, one set per windowsize (with wordLengths[0]) is saved
                //since shortening words to form histograms of different lengths is fast enough
                //and saving EVERY feature set would very quickly eat up a disk
                if (loadFeatureSets) {
                    try {
                        boss = BOSSIndividual.loadFeatureSet(featureFileLoc, data.relationName(), fold, "BOSS", 
                                winSize, wordLengths[0], alphabetSize, normalise);
                    } catch (Exception e){
                        //dont already exist or not found, so make them as usual and serialise them after
                        //todo print to a log file in case of genuine error
                        boss = new BOSSIndividual(wordLengths[0], alphabetSize, winSize, normalise);  
                        boss.buildClassifier(data); //initial setup for this windowsize, with max word length 
                        BOSSIndividual.serialiseFeatureSet(boss, featureFileLoc, data.relationName(), fold);
                    }
                } else {
                    boss = new BOSSIndividual(wordLengths[0], alphabetSize, winSize, normalise);  
                    boss.buildClassifier(data); //initial setup for this windowsize, with max word length     
                }
                
                BOSSIndividual bestClassifierForWinSize = null; 
                double bestAccForWinSize = -1.0;

                //find best word length for this window size
                for (Integer wordLen : wordLengths) {            
                    boss = boss.buildShortenedBags(wordLen); //in first iteration, same lengths (wordLengths[0]), will do nothing
                    
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

                //if this window size's accuracy is not good enough to make it into the ensemble, dont bother storing at all
                if (makesItIntoEnsemble(bestAccForWinSize, maxAcc, minMaxAcc, classifiers.size())) {
                    BOSSWindow bw = new BOSSWindow(bestClassifierForWinSize, bestAccForWinSize, data.relationName());
                    bw.classifier.clean();
                    
                    if (serOption == SerialiseOptions.STORE)
                        bw.store();
                    else if (serOption == SerialiseOptions.STORE_LOAD)
                        bw.storeAndClearClassifier();
                        
                    classifiers.add(bw);
                    
                    if (bestAccForWinSize > maxAcc) {
                        maxAcc = bestAccForWinSize;       
                        //get rid of any extras that dont fall within the new max threshold
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
        
        trainResults.buildTime=System.currentTimeMillis()-trainResults.buildTime;
        if (trainCV) {
            OutFile of=new OutFile(trainCVPath);
            of.writeLine(data.relationName()+",BOSSEnsemble,train");
           
            double[][] results = findEnsembleTrainAcc(data);
            of.writeLine(getParameters());
            of.writeLine(results[0][0]+"");
            ensembleCvAcc = results[0][0];
            for(int i=1;i<results[0].length;i++){
                of.writeString((int)results[0][i]+","+(int)results[1][i]+",");
                for(int j=2;j<results.length;j++)
                    of.writeString(","+results[j][i]);
                of.writeString("\n");
                
            }
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
        
        double[][] results = new double[2+data.numClasses()][data.numInstances() + 1];
        
        this.ensembleCvPreds = new double[data.numInstances()];
        
        double correct = 0; 
        for (int i = 0; i < data.numInstances(); ++i) {
            double[] probs=distributionForInstance(i, data.numClasses());
            double c = 0;
            for(int j=1;j<probs.length;j++)
                if(probs[j]>probs[(int)c])
                    c=j;
                    //No need to do it againclassifyInstance(i, data.numClasses()); //classify series i, while ignoring its corresponding histogram i
            if (c == data.get(i).classValue())
                ++correct;
            results[0][i+1] = data.get(i).classValue();
            results[1][i+1] = c;
            for(int j=0;j<probs.length;j++)   
                results[2+j][i+1]=probs[j];
            this.ensembleCvPreds[i] = c;
        }
        
        results[0][0] = correct / data.numInstances();
        //TODO fill results[1][0]
        
        return results;
    }
    
    public double getEnsembleCvAcc(){
        if(ensembleCvAcc>=0){
            return this.ensembleCvAcc;
        }
        
        try{
            return this.findEnsembleTrainAcc(train)[0][0];
        }catch(Exception e){
            e.printStackTrace();
        }
        return -1;
    }
    
    public double[] getEnsembleCvPreds(){
        if(this.ensembleCvPreds==null){   
            try{
                this.findEnsembleTrainAcc(train);
            }catch(Exception e){
                e.printStackTrace();
            }
        }
        
        return this.ensembleCvPreds;
    }
    
    
    /**
     * Classify the train instance at index 'test', whilst ignoring the corresponding bags 
     * in each of the members of the ensemble, for use in CV of BOSSEnsemble
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
        
        //get votes from all windows 
        double sum = 0;
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
        
        Classifier c = new BOSS();
        c.buildClassifier(train);
        double accuracy = ClassifierTools.accuracy(test, c);
        
        System.out.println("BOSS accuracy on " + dataset + " fold 0 = " + accuracy);
        
        //Other examples/tests
//        detailedFold0Test(dataset);
//        resampleTest(dataset, 100);
    }
    
    public static void detailedFold0Test(String dset) {
        System.out.println("BOSS DetailedTest\n");
        try {
            Instances train = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TRAIN.arff");
            Instances test = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TEST.arff");
            System.out.println(train.relationName());
            
            BOSS boss = new BOSS();
            
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
                System.out.println(i + ": " + params[i][0] + " " + params[i][1] + " " + params[i][2] + " " + 
                        boss.classifiers.get(i).isNorm() + " " + boss.classifiers.get(i).accuracy);
            
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
         
        System.out.println(dset);
        
        Classifier c = new BOSS();
         
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
         
        System.out.println("\n\nBOSSEnsemble mean acc over " + resamples + " resamples: " + mean);
    }
    





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
    public static class BOSSIndividual implements Classifier, Serializable {

        //all sfa words found in original buildClassifier(), no numerosity reduction/shortening applied
        protected BitWord [/*instance*/][/*windowindex*/] SFAwords; 

        //histograms of words of the current wordlength with numerosity reduction applied (if selected)
        public ArrayList<Bag> bags; 

        //breakpoints to be found by MCB
        protected double[/*letterindex*/][/*breakpointsforletter*/] breakpoints;

        public static String classifierName = "BOSS"; //for feature serialistion

        protected double inverseSqrtWindowSize;
        protected int windowSize;
        protected int wordLength;
        protected int alphabetSize;
        protected boolean norm;

        protected boolean numerosityReduction = true; 

        protected static final long serialVersionUID = 1L;

        public BOSSIndividual(int wordLength, int alphabetSize, int windowSize, boolean normalise) {
            this.wordLength = wordLength;
            this.alphabetSize = alphabetSize;
            this.windowSize = windowSize;
            this.inverseSqrtWindowSize = 1.0 / Math.sqrt(windowSize);
            this.norm = normalise;

            //generateAlphabet();
        }

        /**
         * Used when shortening histograms, copies 'meta' data over, but with shorter 
         * word length, actual shortening happens separately
         */
        public BOSSIndividual(BOSSIndividual boss, int wordLength) {
            this.wordLength = wordLength;

            this.windowSize = boss.windowSize;
            this.inverseSqrtWindowSize = boss.inverseSqrtWindowSize;
            this.alphabetSize = boss.alphabetSize;
            this.norm = boss.norm;
            this.numerosityReduction = boss.numerosityReduction; 
            //this.alphabet = boss.alphabet;

            this.SFAwords = boss.SFAwords;
            this.breakpoints = boss.breakpoints;

            bags = new ArrayList<>(boss.bags.size());
        }

        /**
         * Make a complete copy of the passed instance
         * @param boss 
         */
        private BOSSIndividual(BOSSIndividual boss) {
            this.wordLength = boss.wordLength;
            this.windowSize = boss.windowSize;
            this.inverseSqrtWindowSize = boss.inverseSqrtWindowSize;
            this.alphabetSize = boss.alphabetSize;
            this.norm = boss.norm;
            this.numerosityReduction = boss.numerosityReduction; 
            //this.alphabet = boss.alphabet;

            this.SFAwords = boss.SFAwords;
            this.breakpoints = boss.breakpoints;

            this.bags = boss.bags;
        }

        public static class Bag extends HashMap<BitWord, Integer> {
            double classVal;

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

        /**
         * @return { numIntervals(word length), alphabetSize, slidingWindowSize, normalise? } 
         */
        public int[] getParameters() {
            return new int[] { wordLength, alphabetSize, windowSize };
        }

        public void clean() {
            SFAwords = null;
        }

        public static boolean serialiseFeatureSet(BOSSIndividual boss, String path, String dsetName, int fold) {
            path += boss.classifierName+"/"+dsetName+"/"+"fold"+fold+"/";
            File f = new File(path);
            if (!f.exists()) 
                f.mkdirs();

            String filename = boss.classifierName+"_"+dsetName+"_"+fold+"_"+boss.windowSize+"_"+boss.wordLength+"_"+boss.alphabetSize+"_"+boss.norm;

            try {
                ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path + filename));
                out.writeObject(boss);
                out.close();         
                return true;
            }catch(IOException e) {
                System.out.print("Error serialiszing to " + filename);
                e.printStackTrace();
                return false;
            }
        }

        public static BOSSIndividual loadFeatureSet(String path, String dsetName, int fold, String name, 
                int windowSize, int wordLength, int alphabetSize, boolean norm) throws IOException, ClassNotFoundException {
            path += name+"/"+dsetName+"/"+"fold"+fold+"/";

            String filename = name+"_"+dsetName+"_"+fold+"_"+windowSize+"_"+wordLength+"_"+alphabetSize+"_"+norm;
            BOSSIndividual boss = null;
            try {
                ObjectInputStream in = new ObjectInputStream(new FileInputStream(path + filename));
                boss = (BOSSIndividual) in.readObject();
                in.close();
                return boss;
            }catch(IOException i) {
                //System.out.print("Error deserialiszing from " + filename);
                throw i;
            }catch(ClassNotFoundException c) {
                System.out.println("BOSSWindow class not found");
                throw c;
            }
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
            BitWord lastWord = new BitWord();

            for (double[] d : dfts) {
                BitWord word = createWord(d);
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

        protected BitWord createWord(double[] dft) {
            BitWord word = new BitWord(wordLength);
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
        public BOSSIndividual buildShortenedBags(int newWordLength) throws Exception {
            if (newWordLength == wordLength) //case of first iteration of word length search in ensemble
                return this;
            if (newWordLength > wordLength)
                throw new Exception("Cannot incrementally INCREASE word length, current:"+wordLength+", requested:"+newWordLength);
            if (newWordLength < 2)
                throw new Exception("Invalid wordlength requested, current:"+wordLength+", requested:"+newWordLength);

            BOSSIndividual newBoss = new BOSSIndividual(this, newWordLength);

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
        protected Bag createBagFromWords(int thisWordLength, BitWord[] words) {
            Bag bag = new Bag();
            BitWord lastWord = new BitWord();

            for (BitWord w : words) {
                BitWord word = new BitWord(w);
                if (wordLength != thisWordLength)
                    word.shorten(16-thisWordLength); 
                    //TODO hack, word.length=16=maxwordlength, wordLength of 'this' BOSS instance unreliable, length of SFAwords = maxlength

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

        protected BitWord[] createSFAwords(Instance inst) throws Exception {            
            double[][] dfts = performMFT(toArrayNoClass(inst)); //approximation     
            BitWord[] words = new BitWord[dfts.length];
            for (int window = 0; window < dfts.length; ++window) 
                words[window] = createWord(dfts[window]);//discretisation

            return words;
        }

        @Override
        public void buildClassifier(Instances data) throws Exception {
            
            if (data.classIndex() != data.numAttributes()-1)
                throw new Exception("BOSS_BuildClassifier: Class attribute not set as last attribute in dataset");

            breakpoints = MCB(data); //breakpoints to be used for making sfa words for train AND test data

            SFAwords = new BitWord[data.numInstances()][];
            bags = new ArrayList<>(data.numInstances());

            for (int inst = 0; inst < data.numInstances(); ++inst) {
                SFAwords[inst] = createSFAwords(data.get(inst));

                Bag bag = createBagFromWords(wordLength, SFAwords[inst]);
                bag.setClassVal(data.get(inst).classValue());
                bags.add(bag);
            }
            
        }

        /**
         * Computes BOSS distance between two bags d(test, train), is NON-SYMETRIC operation, ie d(a,b) != d(b,a)
         * @return squared distance FROM instA TO instB
         */
        public double BOSSdistance(Bag instA, Bag instB) {
            double dist = 0.0;

            //find dist only from values in instA
            for (Entry<BitWord, Integer> entry : instA.entrySet()) {
                Integer valA = entry.getValue();
                Integer valB = instB.get(entry.getKey());
                if (valB == null)
                    valB = 0;
                dist += (valA-valB)*(valA-valB);
            }

            return dist;
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
            for (Entry<BitWord, Integer> entry : instA.entrySet()) {
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
        public double classifyInstance(Instance instance) throws Exception {
            Bag testBag = BOSSTransform(instance);

            double bestDist = Double.MAX_VALUE;
            double nn = -1.0;

            //find dist FROM testBag TO all trainBags
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
         * @param test index of instance to classify
         * @return classification
         */
        public double classifyInstance(int test) {

            double bestDist = Double.MAX_VALUE;
            double nn = -1.0;

            Bag testBag = bags.get(test);

            for (int i = 0; i < bags.size(); ++i) {
                if (i == test) //skip 'this' one, leave-one-out
                    continue;

                double dist = BOSSdistance(testBag, bags.get(i), bestDist); 

                if (dist < bestDist) {
                    bestDist = dist;
                    nn = bags.get(i).getClassVal();
                }
            }

            return nn;
        }

        @Override
        public double[] distributionForInstance(Instance instance) throws Exception {
            //TODO implement
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public Capabilities getCapabilities() {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        public static void detailedFold0Test(String dset) {
            System.out.println("BOSSIndividual DetailedTest\n");
            try {
                Instances train = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TRAIN.arff");
                Instances test = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TEST.arff");
                System.out.println(train.relationName());

                int windowSize = 10; 
                int alphabetSize = 4;
                int wordLength = 43;
                boolean norm = true;

                BOSSIndividual boss = new BOSSIndividual(windowSize, alphabetSize, wordLength, norm);
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