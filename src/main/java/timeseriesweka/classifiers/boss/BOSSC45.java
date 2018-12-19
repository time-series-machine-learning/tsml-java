package timeseriesweka.classifiers.boss;

import fileIO.OutFile;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream; 
import java.io.IOException; 
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import weka.core.Capabilities;
import weka.classifiers.Classifier;
import weka.core.TechnicalInformation;

import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Set;
import timeseriesweka.classifiers.BOSS;
import utilities.BitWord;
import utilities.ClassifierTools;
import utilities.TrainAccuracyEstimate;
import utilities.ClassifierResults;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
 
/**
 * BOSS + C45 tree classifier with parameter search and ensembling, if parameters are 
 * known, use nested 'BOSSC45Individual' classifier and directly provide them.
 * 
 * Intended use is with the default constructor, however can force the normalisation 
 * parameter to true/false by passing a boolean, e.g c = new BOSSEnsemble(true)
 * 
 * Alphabetsize fixed to four
 * 
 * @author James Large
 * 
 * 
 * TEMPORARY WORK AROUND, CONSULT TODO: test cases may contain words not found anywhere 
 * in the train set, and therefore not found in the dictionary (i.e list of attributes
 * in the instance/s). Currently words found in test instances that are not in the train set are 
 * IGNORED. 
 * 
 * Largely poor accuracy likely due to this
 * 
 * BOSS implementation based on the algorithm described in getTechnicalInformation()
 * C45 done using the WEKA implementation 'weka.classifiers.trees.J48'
 */
public class BOSSC45 implements Classifier, SaveParameterInfo,TrainAccuracyEstimate {
    
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

    private int numCVFolds = 10;
    
    private final double correctThreshold = 0.92;
    private int maxEnsembleSize = 50;
    
    private final Integer[] wordLengths = { 16, 14, 12, 10, 8 };
    private final int alphabetSize = 4;
    //private boolean norm;
    
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
    public BOSSC45(boolean normalise) {
        normOptions = new boolean[] { normalise };
    }
    
    /**
     * During buildClassifier(...), will search through normalisation as well as 
     * window size and word length if no particular normalisation option is provided
     */
    public BOSSC45() {
        normOptions = new boolean[] { true, false };
    }  

    public static class BOSSWindow implements Comparable<BOSSWindow>, Serializable { 
        private BOSSC45Individual classifier;
        public double accuracy;
        public String filename;
        
        private static final long serialVersionUID = 2L;

        public BOSSWindow(String filename) {
            this.filename = filename;
        }
        
        public BOSSWindow(BOSSC45Individual classifer, double accuracy, String dataset) {
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
            filename = serFileLoc + dataset + "_" + classifier.getWindowSize() + "_" + classifier.getWordLength() + "_" + classifier.getAlphabetSize() + "_" + classifier.isNorm() + ".ser";
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
            throw new Exception("BOSSEnsemble_BuildClassifier: Class attribute not set as last attribute in dataset");
 
        if (serOption == SerialiseOptions.STORE || serOption == SerialiseOptions.STORE_LOAD) {
            DateFormat dateFormat = new SimpleDateFormat("yyyyMMddHHmmss");
            Date date = new Date();
            serFileLoc += data.relationName() + "_" + dateFormat.format(date) + "\\";
            File f = new File(serFileLoc);
            if (!f.isDirectory())
                f.mkdirs();
        }
        
        
        if (data.numInstances() < numCVFolds)
            numCVFolds = data.numInstances();
        
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
        double minMaxAcc = -1.0; //the acc of the worst member to make it into the final ensemble as it stands
        
        for (boolean normalise : normOptions) {
            for (int winSize = minWindow; winSize <= maxWindow; winSize += winInc) {          
                BOSSC45Individual boss = new BOSSC45Individual(wordLengths[0], alphabetSize, winSize, normalise); 
                boss.buildBags=true;
                boss.buildForest=false;
                boss.buildClassifier(data); //initial setup for this windowsize, with max word length     

                BOSSC45Individual bestClassifierForWinSize = null; 
                double bestAccForWinSize = -1.0;

                //find best word length for this window size
                for (Integer wordLen : wordLengths) {            
                    boss = boss.buildShortenedBags(wordLen); //in first iteration, same lengths (wordLengths[0]), will do nothing
                    
                    double acc = boss.findCVAcc(numCVFolds);
                    
                    if (acc >= bestAccForWinSize) {
                        bestAccForWinSize = acc;
                        bestClassifierForWinSize = boss;
                    }
                }

                //if not within correct threshold of the current max, dont bother storing at all
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
        
        //final members known, now build the final trees on the full train data (rather than train/validation set)
        for (BOSSWindow window : classifiers)
            window.classifier.buildFullForest();
        
        if (trainCV) {
            OutFile of=new OutFile(trainCVPath);
            of.writeLine(data.relationName()+",BOSSEnsemble,train");
           
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
        String dataset = "BeetleFly";
        Instances train = ClassifierTools.loadData("C:\\TSC Problems\\"+dataset+"\\"+dataset+"_TRAIN.arff");
        Instances test = ClassifierTools.loadData("C:\\TSC Problems\\"+dataset+"\\"+dataset+"_TEST.arff");
        
        Classifier c = new BOSSC45();
        c.buildClassifier(train);
        double accuracy = ClassifierTools.accuracy(test, c);
        
        System.out.println("BOSSC45 accuracy on " + dataset + " fold 0 = " + accuracy);
        
        //Other examples/tests
//        detailedFold0Test(dataset);
//        resampleTest(dataset, 5);
    }
    
    public static void detailedFold0Test(String dset) {
        System.out.println("BOSSC45 DetailedTest\n");
        try {
            Instances train = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TRAIN.arff");
            Instances test = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TEST.arff");
            System.out.println(train.relationName());
            
            BOSSC45 boss = new BOSSC45();
            
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
                System.out.println(i + ": " + params[i][0] + " " + params[i][1] + " " + params[i][2] + " " + boss.classifiers.get(i).isNorm() + " " + boss.classifiers.get(i).accuracy);
            
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
         
        Classifier c = new BOSSC45();
         
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
     * TEMPORARY WORK AROUND, CONSULT TODO: test cases may contain words not found anywhere 
     * in the train set, and therefore not found in the dictionary (i.e list of attributes
     * in the instance/s). Currently words found in test instances that are not in the train set are 
     * IGNORED. 
     * 
     * Largely poor accuracy likely due to this
     * 
     * @author James Large
     */
    public static class BOSSC45Individual extends BOSS.BOSSIndividual {

        public boolean buildBags = true;
        public boolean buildForest = true;

        public Instances bagInsts;
        public J48 tree = new J48();
        private Attribute classAttribute;

        public BOSSC45Individual(int wordLength, int alphabetSize, int windowSize, boolean normalise) {
            super(wordLength, alphabetSize, windowSize, normalise);
        }

        public BOSSC45Individual(BOSSC45Individual boss, int wordLength) {
            super(boss, wordLength);

            classAttribute = boss.classAttribute;
        }

        public Instances bagsToInstances() {
            //build attribute info, each bag may have different keys etc
            //need to build common vector of all keys found in all bags
            //this is the main source of memory problems, most bags will have many unique
            //keys of value 1 due to noise, thus FULL keyset is much larger than 
            //each individuals bag's keyset
            FastVector<Attribute> attInfo = new FastVector<>();
            Set<String> wordsFound = new HashSet<>();
            for (Bag bag : bags) 
                for (Entry<BitWord,Integer> entry : bag.entrySet()) 
                    wordsFound.add(entry.getKey().toString());
            for (String word : wordsFound) 
                attInfo.add(new Attribute(word));

            //classvals must not be numeric...            
            attInfo.add(classAttribute);

            //atts found, now populate all values
            Instances bagInsts = new Instances("", attInfo, bags.size());
            bagInsts.setClassIndex(attInfo.size()-1);

            int i = 0;
            for (Bag bag : bags) {
                //init all values to 0, + class value at the end
                double[] init = new double[attInfo.size()];
                init[init.length-1] = bag.getClassVal();

                bagInsts.add(new DenseInstance(1, init));
                for (Entry<BitWord,Integer> entry : bag.entrySet())
                    bagInsts.get(i).setValue(bagInsts.attribute(entry.getKey().toString()), entry.getValue());

                i++;
            }

            return bagInsts;
        }

        @Override
        public void buildClassifier(Instances data) throws Exception {
            classAttribute = data.classAttribute();

            if (buildBags)
                super.buildClassifier(data);

            if (buildForest)
                buildFullForest();
            //accomodates some speed ups when using cv in ensemble
        }

        public void buildFullForest() throws Exception {
            //first pass, building normal bags (ArrayList<Bag>) 
            //then converting to instances for use in rotf
            //TODO potentially build bags straight into instances form, (mem/very slight speed saving) 
            //however the mem savings are made anyway when clean() is called after training
            //and would require a big overhaul
            if (bagInsts == null) 
                bagInsts = bagsToInstances();

            if (tree == null) 
                tree = new J48();
            tree.buildClassifier(bagInsts);
        }

        public double findCVAcc(int numFolds) throws Exception {
            if (bagInsts == null) 
                bagInsts = bagsToInstances();
            if (tree == null) 
                tree = new J48();
            return ClassifierTools.crossValidationWithStats(tree, bagInsts, numFolds)[0][0];
        }

        @Override
        public BOSSC45Individual buildShortenedBags(int newWordLength) throws Exception {
            if (newWordLength == wordLength) //case of first iteration of word length search in ensemble
                return this;
            if (newWordLength > wordLength)
                throw new Exception("Cannot incrementally INCREASE word length, current:"+wordLength+", requested:"+newWordLength);
            if (newWordLength < 2)
                throw new Exception("Invalid wordlength requested, current:"+wordLength+", requested:"+newWordLength);

            BOSSC45Individual newBoss = new BOSSC45Individual(this, newWordLength);

            //build hists with new word length from SFA words, and copy over the class values of original insts
            for (int i = 0; i < bags.size(); ++i) {
                Bag newBag = createBagFromWords(newWordLength, SFAwords[i]);   
                newBag.setClassVal(bags.get(i).getClassVal());
                newBoss.bags.add(newBag);
            }

            return newBoss;
        }

        @Override
        public void clean() {
            //null out things that are not needed after training to save memory
            super.clean();
            bags = null;
            tree = null;
            //bagInsts = null; //needed to build full forest if this makes it into the final ensemble
        }

        @Override
        public double classifyInstance(Instance instance) throws Exception {
            Bag testBag = BOSSTransform(instance);

            //convert bag to instance
            double[] init = new double[bagInsts.numAttributes()];
            init[init.length-1] = testBag.getClassVal();

            //TEMPORARILY create it on the end of the train insts to easily copy over the attribute data.
            bagInsts.add(new DenseInstance(1, init));
            for (Entry<BitWord,Integer> entry : testBag.entrySet()) {
                Attribute att = bagInsts.attribute(entry.getKey().toString());
                if (att != null)
                    bagInsts.get(bagInsts.size()-1).setValue(att, entry.getValue());
            }

            Instance testInst = bagInsts.remove(bagInsts.size()-1);

            return tree.classifyInstance(testInst);
        }

        //hacky work around to allow for use of ClassifierTools.
        @Override
        public double[] distributionForInstance(Instance instance) throws Exception {
            Bag testBag = BOSSTransform(instance);

            //convert bag to instance
            double[] init = new double[bagInsts.numAttributes()];
            init[init.length-1] = testBag.getClassVal();

            //TEMPORARILY create it on the end of the train isnts to easily copy over the attribute data.
            bagInsts.add(new DenseInstance(1, init));
            for (Entry<BitWord,Integer> entry : testBag.entrySet()) {
                Attribute att = bagInsts.attribute(entry.getKey().toString());
                if (att != null)
                    bagInsts.get(bagInsts.numInstances()-1).setValue(att, entry.getValue());
            }
            Instance testInst = bagInsts.remove(bagInsts.size()-1);

            return tree.distributionForInstance(testInst);
        }
    }

}