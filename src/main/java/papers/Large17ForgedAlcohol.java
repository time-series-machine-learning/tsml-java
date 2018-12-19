package papers;

import fileIO.OutFile;
import java.io.File;
import java.util.Random;
import timeseriesweka.classifiers.BOSS;
import timeseriesweka.classifiers.ParameterSplittable;
import timeseriesweka.classifiers.RISE;
import timeseriesweka.classifiers.TSF;
import utilities.ClassifierTools;
import utilities.CrossValidator;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.meta.RotationForest;
import utilities.ClassifierResults;
import vector_classifiers.CAWPE;
import timeseriesweka.classifiers.ensembles.SaveableEnsemble;
import vector_classifiers.PLSNominalClassifier;
import vector_classifiers.SaveEachParameter;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.kNN;
import weka.core.Instances;
import development.DataSets;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeSet;
import weka.attributeSelection.PrincipalComponents;
import weka.core.Attribute;
import weka.core.Instance;

/**
 * Accompanying code for the paper titled
 * 'Detecting forged alcohol non-invasively through vibrational spectroscopy and machine learning'.
 * 
 * Main port of call is the method paperExampleCode()
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class Large17ForgedAlcohol{
    public static int folds=30; 
    static int numCVFolds = 10;
    static boolean debug=true;
    static boolean checkpoint=false;
    static boolean generateTrainFiles=true;
    static Integer parameterNum=0;
    static double trainSplitProportion = 0.7;
    
    static String loboFolder = "LOBO/";
    static String randBottleFolder = "RandomBottles/";
    static String randForgeryFolder = "RandomForgery/";
    
    public static Classifier setClassifier(String classifier, int fold) throws Exception{
        Classifier c=null;
        SMO smo=null;
        switch(classifier){      
            case "1NN":
                c = new kNN();
                break;
            case "C45":
                c=new J48();
                break;
            case "PLSNominalClassifier":
                c = new PLSNominalClassifier();
                break;                
            case "SVML":
                smo = new SMO();
                smo.turnChecksOff();
                smo.setBuildLogisticModels(true);
                PolyKernel kl = new PolyKernel();
                kl.setExponent(1);
                smo.setKernel(kl);
                smo.setRandomSeed(fold);
                c = smo;
                break;
            case "SVMQ":
                smo = new SMO();
                smo.turnChecksOff();
                smo.setBuildLogisticModels(true);
                PolyKernel kq = new PolyKernel();
                kq.setExponent(2);
                smo.setKernel(kq);
                smo.setRandomSeed(fold);
                c = smo;
                break;
            case "SVMRBF":
                smo = new SMO();
                smo.turnChecksOff();
                smo.setBuildLogisticModels(true);
                RBFKernel rbf = new RBFKernel();
                smo.setKernel(rbf);
                smo.setRandomSeed(fold);
                c = smo;
                break;
            case "RandF":
                RandomForest r=new RandomForest();
                r.setNumTrees(500);
                r.setSeed(fold);            
                c = r;
                break;
            case "RotF":
                RotationForest rf=new RotationForest();
                rf.setNumIterations(50);
                rf.setSeed(fold);
                c = rf;
                break;
            case "MLP":
                c=new MultilayerPerceptron();
                break;
            case "Logistic":
                c= new Logistic();
                break;
            case "HESCA":
                c=new CAWPE();
                ((CAWPE)c).setRandSeed(fold);
                break;
            case "TSF":
                c=new TSF();
                break;
            case "RISE":
                c=new RISE();
                break;
            case "BOSS": case "BOSSEnsemble": 
                c=new BOSS();
                break;
            default:
               throw new Exception("Unknown classifier: " + classifier);
        }
        return c;
    }
        
/* MUST BE at least Arguments:
    1: Problem path args[0]
    2. Results path args[1]
    3. booleanwWhether to CV to generate train files (true/false)
    4. Classifier =args[3];
    5. String problem=args[4];
    6. int fold=Integer.parseInt(args[5])-1;
Optional    
    7. boolean whether to checkpoint parameter search (true/false)
    8. integer for specific parameter search (0 indicates ignore this) 
    */  
       
    public static void main(String[] args) throws Exception{
        if (args.length > 0) {
            clusterRun(args);
            return;
        }
        
        paperExampleCode();
    }
    
    public static Instances[] sampleFromPredefinedFolds(String problem, int fold) {
        Instances train = ClassifierTools.loadData(problem+fold+"_TRAIN");
        Instances test = ClassifierTools.loadData(problem+fold+"_TEST");
        return new Instances[] { train, test };
    }
    
    /**
     * Assumes that the full dataset *WITH THE BOTTLE ATTRIBUTE* are passed
     */
    public static Instances[] sampleFromFullDataset(String dataset, int fold, String method) throws Exception{
        Instances data = null;
        Instances[] split = null;
        
        Instances train, test;
        
        switch (method) {
            case "LOBO":
                data=ClassifierTools.loadData(DataSets.problemPath+dataset); 
                split=resampleLOBOFromInstances(data, fold, true);
                break;  
            case "LOBOPCA":
                data=ClassifierTools.loadData(DataSets.problemPath+dataset); 
                split=resampleLOBOFromInstances(data, fold, true);
                
                PrincipalComponents pca = new PrincipalComponents();
                pca.setVarianceCovered(0.95);
                pca.buildEvaluator(split[0]);
                Instances pcaTrain = pca.transformedData(split[0]);

                Instances pcaTest = new Instances(pca.transformedHeader());
                for (Instance instance : split[1])
                    pcaTest.add(pca.convertInstance(instance));
                split = new Instances[] { pcaTrain, pcaTest };
                break;
            case "RandomBottles":
                data=ClassifierTools.loadData(DataSets.problemPath+dataset); 
                split = resampleBottleClassFromInstances(data, fold);
                break;
            default: 
                System.out.println("Unrecognised sampling method for alcohol dsets, SpectralDataResultsFileBuilder.sample(...): " + method);
                System.exit(0);
        }
        
        
        return split;
    }
    
    public static Instances[] resampleBottleClassFromInstances(Instances all2, int seed) {       
        Random rand = new Random(seed);
        
        Instances allData = new Instances(all2);
        allData.setClassIndex(2); //temporary
        
        Attribute newClassAtt = allData.attribute(0);
        double[] newClassVals = allData.attributeToDoubleArray(0);
        allData.deleteAttributeAt(0);
        allData.deleteAttributeAt(allData.numAttributes()-1);
        
        allData.insertAttributeAt(newClassAtt, allData.numAttributes());
        allData.setClassIndex(allData.numAttributes()-1);
        
        for (int i = 0; i < allData.numInstances(); i++)
            allData.instance(i).setValue(allData.numAttributes()-1, newClassVals[i]);
        
        return InstanceTools.resampleInstances(allData, seed, trainSplitProportion);
    }
    
    public static Instances[] resampleLOBOFromInstances(Instances all2, int seed, boolean removeBottleAtt) {       
        Instances all = new Instances(all2);
        
        Set<Double> bottleSet = new TreeSet<>();
        for (Instance inst : all)
            bottleSet.add(inst.value(0));
        
        int numBottles = bottleSet.size();
        int samplesPerBottle = all.numInstances() / numBottles;
        double testBottle = (double) (seed % numBottles);
        
        Instances[] data = new Instances[2];
        data[0] = new Instances(all, (numBottles-1) * samplesPerBottle);
        data[1] = new Instances(all, samplesPerBottle);
        
        Iterator<Instance> iter = all.iterator();
        while (iter.hasNext()) {
            Instance inst = iter.next();
            if (inst.value(0) == testBottle) {
                data[1].add(inst);
                iter.remove();
            }
               // data[1].add(train.removeIf(new Predicate<Instance>(Instance inst) { return p -> inst.value(0) == testBottle }));
        }
        
        data[0].addAll(all);
            
        Random rand = new Random(seed);
        data[0].randomize(rand);
        data[1].randomize(rand);
        
        if (removeBottleAtt)
            removeBottleAttribute(data);
        
        return data;
    }
    
    public static void removeBottleAttribute(Instances data) {
        if (data.attribute("bottleName") != null)
            data.deleteAttributeAt(data.attribute("bottleName").index());
    }
    
    public static void removeBottleAttribute(Instances[] data) {
        if (data[0].attribute("bottleName") != null)
            data[0].deleteAttributeAt(data[0].attribute("bottleName").index());
        if (data[1].attribute("bottleName") != null)
            data[1].deleteAttributeAt(data[1].attribute("bottleName").index());
    }
    
    public static void paperExampleCode() throws Exception {
        //Would perform Leave-one-bottle-out experiments. 
        //Intended as example code for the experimental procedure, in reality 
        //experiments were distributed on the UEA's HPC cluster (using a version of 
        //the code moved into clusterRun(...)) 
        
        //BUG: 
        //Post submission, it was found that there is a double precision error, resulting 
        //in (on the order of) one in every few thousand instances being classified differently
        //Here, train/test splits are defined by loading in the full dataset and splitting 
        //in memory. In the experiments that made the published results, splits were locally 
        //made and saved to file, before being read in to classify on the cluster. 
        //The method used to save the splits wrote out to 6 decimal places, whereas creating them in memory
        //will mean they have full double precision. 
        //On average, results should be identical, if anything the extra precision should 
        //mean results created here are on average a bit better, though no where near 
        //significantly so. 
        
        System.out.println("Performing Leave-one-bottle-out evaluations for ethanol concentration determination");
        
        //change these paths to where you have downloaded the data
        boolean useSavedSplits = true;
        String baseDataPath = "";
        if (useSavedSplits) 
            baseDataPath = "Z:/Data/AlcoholForgeryFolds/";
        else 
            baseDataPath = "Z:/Data/AlcoholProblems/AllDataWithBottleAttribute/";
        
        
        String[] samplingMethods = { "LOBO", "LOBOPCA", "RandomBottle" };
        int numFolds = 44;
        String [] classifiers = {"C45"};//,"RotF","RandF","SVML","SVMQ","SVMRBF","CAWPE","RISE","BOSS","TSF","Logistic","MLP","1NN","PLSNominalClassifier"};
        String[] datasets = { "AlcoholForgeryEthanol", "AlcoholForgeryMethanol" };
        
        for (String samplingMethod : samplingMethods) {
            System.out.println("Sampling method: " + samplingMethod);
            
            String PCAprefix = samplingMethod.contains("PCA") ? "PCA" : "";
            for (String dataset : datasets) {            
                System.out.println("\tDataset: " + dataset);
                dataset = PCAprefix + dataset;
                    
                DataSets.problemPath = baseDataPath + dataset + "/";
                
                for (String classifier : classifiers) {
                    System.out.println("\t\tClassifier: " + classifier);
                    double acc = .0;

                    for (int fold = 0; fold < numFolds; fold++) {
                        Instances[] split = null;
                        if (useSavedSplits) 
                            split = sampleFromPredefinedFolds(baseDataPath+samplingMethod+"/"+dataset+"/"+dataset, fold); //includes precision error, and allows precise recreation
                        else 
                            split = sampleFromFullDataset(dataset, fold, samplingMethod); //excludes precision error, produces insignificantly different average results

                        Classifier c = setClassifier(classifier, fold);                
                        c.buildClassifier(split[0]);

                        double foldAcc = ClassifierTools.accuracy(split[1], c);
                        System.out.println("\t\t\tFold" + fold + " acc: " + foldAcc);
                        acc += foldAcc;
                    }

                    acc /= numFolds;
                    System.out.println("\t\t" + classifier + " average accuracy: " + acc);
                }
            }
        }
    }
    
    public static void clusterRun(String[] args) throws Exception {
        for(String str:args)
            System.out.println(str);
         
        DataSets.problemPath=args[0];
        DataSets.resultsPath=args[1];
//Arg 3 argument is whether to cross validate or not and produce train files
        generateTrainFiles=Boolean.parseBoolean(args[2]);
        File f=new File(DataSets.resultsPath);
        if(!f.isDirectory()){
            f.mkdirs();
        }
// Arg 4,5,6 Classifier, Problem, Fold             
        String[] newArgs=new String[3];
        for(int i=0;i<3;i++)
            newArgs[i]=args[i+3];
//OPTIONAL
//  Arg 7:  whether to checkpoint        
        checkpoint=false;
        if(args.length>=7){
            String s=args[args.length-1].toLowerCase();
            if(s.equals("true"))
                checkpoint=true;
        }
//Arg 8: if present, do a single parameter split
        parameterNum=0;
        if(args.length>=8)
            parameterNum=Integer.parseInt(args[7]);
        
        System.out.println("Checkpoint ="+checkpoint+" param number ="+parameterNum);
        singleClassifierAndFoldTrainTestSplit(newArgs);
    }
    
    /** Run a given classifier/problem/fold combination with associated file set up
 @param args: 
 * args[0]: Classifier name. Create classifier with setClassifier
 * args[1]: Problem name
 * args[2]: Fold number. This is assumed to range from 1, hence we subtract 1
 * (this is because of the scripting we use to run the code on the cluster)
 *          the standard archive folds are always fold 0
 * 
 * NOTES: 
 * 1. this assumes you have set DataSets.problemPath to be where ever the 
 * data is, and assumes the data is in its own directory with two files, 
 * args[1]_TRAIN.arff and args[1]_TEST.arff 
 * 2. assumes you have set DataSets.resultsPath to where you want the results to
 * go It will NOT overwrite any existing results (i.e. if a file of non zero 
 * size exists)
 * 3. This method just does the file set up then calls the next method. If you 
 * just want to run the problem, go to the next method
* */
    public static void singleClassifierAndFoldTrainTestSplit(String[] args) throws Exception{
//first gives the problem file      
        String classifier=args[0];
        String problem=args[1];
        int fold=Integer.parseInt(args[2])-1;
   
        String predictions = DataSets.resultsPath+classifier+"/Predictions/"+problem;
        File f=new File(predictions);
        if(!f.exists())
            f.mkdirs();
        
        //Check whether fold already exists, if so, dont do it, just quit
        f=new File(predictions+"/testFold"+fold+".csv");
        if(!f.exists() || f.length()==0){
            Classifier c = setClassifier(classifier,fold);
            
            //Sample the dataset
            Instances[] data = sampleFromPredefinedFolds(DataSets.problemPath+problem+"/"+problem, fold);
            
            if (c instanceof PLSNominalClassifier) {
                if (data[0].numAttributes() < ((PLSNominalClassifier)c).getNumComponents()) //default 20, don't know enough to want to change it
                    ((PLSNominalClassifier)c).setNumComponents(data[0].numAttributes() - 1); //however must have at least as many attributes as components
            }                   
            
            if(parameterNum>0 && c instanceof ParameterSplittable)//Single parameter fold
            {
                checkpoint=false;
//Check if it already exists, if it does, exit
                f=new File(predictions+"/fold"+fold+"_"+parameterNum+".csv");
                if(f.exists() && f.length()>0){ //Exit
                    System.out.println("Fold "+predictions+"/fold"+fold+"_"+parameterNum+".csv  already exists");
                    return; //Aready done
                }
            }
            
            double acc = singleClassifierAndFoldTrainTestSplit(data[0],data[1],c,fold,predictions);
            System.out.println(classifier+","+problem+","+fold+","+acc);
        }
    }
/**
 * 
 * @param train: the standard train fold Instances from the archive 
 * @param test: the standard test fold Instances from the archive
 * @param c: Classifier to evaluate
 * @param fold: integer to indicate which fold. Set to 0 to just use train/test
 * @param resultsPath: a string indicating where to store the results
 * @return the accuracy of c on fold for problem given in train/test
 * 
 * NOTES:
 * 1.  If the classifier is a SaveableEnsemble, then we save the internal cross 
 * validation accuracy and the internal test predictions
 * 2. The output of the file testFold+fold+.csv is
 * Line 1: ProblemName,ClassifierName, train/test
 * Line 2: parameter information for final classifier, if it is available
 * Line 3: test accuracy
 * then each line is
 * Actual Class, Predicted Class, Class probabilities 
 * 
 * 
 */    
    public static double singleClassifierAndFoldTrainTestSplit(Instances train, Instances test, Classifier c, int fold,String resultsPath){
        String testFoldPath="/testFold"+fold+".csv";
        String trainFoldPath="/trainFold"+fold+".csv";
        
        ClassifierResults trainResults = null;
        ClassifierResults testResults = null;
        
        if(parameterNum>0 && c instanceof ParameterSplittable)//Single parameter fold
        {
            checkpoint=false;
            ((ParameterSplittable)c).setParametersFromIndex(parameterNum);
//            System.out.println("classifier paras =");
            testFoldPath="/fold"+fold+"_"+parameterNum+".csv";
            generateTrainFiles=false;
        }
        else{
//Only do all this if not an internal fold
    // Save internal info for ensembles
            if(c instanceof SaveableEnsemble)
               ((SaveableEnsemble)c).saveResults(resultsPath+"/internalCV_"+fold+".csv",resultsPath+"/internalTestPreds_"+fold+".csv");
            if(checkpoint && c instanceof SaveEachParameter){     
                ((SaveEachParameter) c).setPathToSaveParameters(resultsPath+"/fold"+fold+"_");
            }
        }
        
        try{             
            if(generateTrainFiles){
                if(c instanceof TrainAccuracyEstimate) //Classifier will perform cv internally
                    ((TrainAccuracyEstimate)c).writeCVTrainToFile(resultsPath+trainFoldPath);
                else{ // Need to cross validate here
                    int numFolds = Math.min(train.numInstances(), numCVFolds);
                    CrossValidator cv = new CrossValidator();
                    cv.setSeed(fold);
                    cv.setNumFolds(numFolds);
                    trainResults=cv.crossValidateWithStats(c,train);
                }
            }
            
            //Build on the full train data here
            long buildTime=System.currentTimeMillis();
            c.buildClassifier(train);
            buildTime=System.currentTimeMillis()-buildTime;
            
            if (generateTrainFiles) { //And actually write the full train results if needed
                if(!(c instanceof TrainAccuracyEstimate)){ 
                    OutFile trainOut=new OutFile(resultsPath+trainFoldPath);
                    trainOut.writeLine(train.relationName()+","+c.getClass().getName()+",train");
                    if(c instanceof SaveParameterInfo )
                        trainOut.writeLine(((SaveParameterInfo)c).getParameters()); //assumes build time is in it's param info, is for tunedsvm
                    else 
                        trainOut.writeLine("BuildTime,"+buildTime+",No Parameter Info");
                    trainOut.writeLine(trainResults.acc+"");
                    trainOut.writeLine(trainResults.writeInstancePredictions());
                    //not simply calling trainResults.writeResultsFileToString() since it looks like those that extend SaveParameterInfo will store buildtimes
                    //as part of their params, and so would be written twice
                    trainOut.closeFile();
                }
            }
            
            //Start of testing
            int numInsts = test.numInstances();
            int pred;
            testResults = new ClassifierResults(test.numClasses());
            double[] trueClassValues = test.attributeToDoubleArray(test.classIndex()); //store class values here
            
            for(int testInstIndex = 0; testInstIndex < numInsts; testInstIndex++) {
                test.instance(testInstIndex).setClassMissing();//and remove from each instance given to the classifier (just to be sure)
                
                //make prediction
                double[] probs=c.distributionForInstance(test.instance(testInstIndex));
                testResults.storeSingleResult(probs);
            }
            testResults.finaliseResults(trueClassValues); 
            
            //Write results
            OutFile testOut=new OutFile(resultsPath+testFoldPath);
            testOut.writeLine(test.relationName()+","+c.getClass().getName()+",test");
            if(c instanceof SaveParameterInfo)
              testOut.writeLine(((SaveParameterInfo)c).getParameters());
            else
                testOut.writeLine("No parameter info");
            testOut.writeLine(testResults.acc+"");
            testOut.writeString(testResults.writeInstancePredictions());
            testOut.closeFile();
            
            return testResults.acc;
        } catch(Exception e) {
            System.out.println(" Error ="+e+" in method simpleExperiment"+e);
            e.printStackTrace();
            System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
            System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes"+test.numInstances()+" instances");

            return Double.NaN;
        }
    }
}