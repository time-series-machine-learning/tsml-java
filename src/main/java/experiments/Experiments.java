/**
 *
 * @author ajb
 *local class to run experiments with the UCR-UEA or UCI data


*/
package experiments;

import fileIO.OutFile;
import java.io.File;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import multivariate_timeseriesweka.classifiers.*;
import timeseriesweka.classifiers.BOSS;
import timeseriesweka.classifiers.BagOfPatterns;
import timeseriesweka.classifiers.DD_DTW;
import timeseriesweka.classifiers.DTD_C;
import timeseriesweka.classifiers.ElasticEnsemble;
import timeseriesweka.classifiers.FastShapelets;
import timeseriesweka.classifiers.FlatCote;
import timeseriesweka.classifiers.HiveCote;
import timeseriesweka.classifiers.LPS;
import timeseriesweka.classifiers.LearnShapelets;
import timeseriesweka.classifiers.NN_CID;
import timeseriesweka.classifiers.ParameterSplittable;
import timeseriesweka.classifiers.RISE;
import timeseriesweka.classifiers.SAXVSM;
import timeseriesweka.classifiers.ShapeletTransformClassifier;
import timeseriesweka.classifiers.SlowDTW_1NN;
import timeseriesweka.classifiers.TSBF;
import timeseriesweka.classifiers.TSF;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.DTW1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.ED1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.MSM1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.WDTW1NN;
import utilities.ClassifierTools;
import utilities.CrossValidator;
import utilities.InstanceTools;
import timeseriesweka.classifiers.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.meta.RotationForest;
import utilities.ClassifierResults;
import vector_classifiers.CAWPE;
import timeseriesweka.classifiers.ensembles.SaveableEnsemble;
import timeseriesweka.classifiers.FastWWS.FastDTWWrapper;
import timeseriesweka.classifiers.WEASEL;
import utilities.multivariate_tools.MultivariateInstanceTools;
import vector_classifiers.*;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomTree;
import weka.core.Attribute;
import weka.core.EuclideanDistance;
import weka.core.Instances;


    

public class Experiments implements Runnable{
//For threaded version
    String[] args;
    public static int folds=30; 
    static int numCVFolds = 10;
    static boolean debug=true;
    static boolean checkpoint=false;
    static boolean generateTrainFiles=true;
    static Integer parameterNum=0;
    static boolean singleFile=false;
    static boolean foldsInFile=false;
    static boolean useBagsSampling=false;//todo is a hack for bags project experiments 
    static double SPLITPROP=0.5;    
    public static String threadClassifier="ED";    
    public static String[] cmpv2264419={
    "adult",
    };
    public static String[] ajb17pc={
    "abalone"
    };    
    public static String[] cmpv2202398={
    "adult"
    };
    //TODO
    /*

    */
    public static String[] laptop={
    "balloons"  
    };
/** This method is now too bloated
 * 
 * @param classifier
 * @param fold
 * @return 
 */    
    public static Classifier setClassifier(String classifier, int fold){
        Classifier c=null;
        switch(classifier){
            case "ContractRotationForest":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setDayLimit(5);
                ((ContractRotationForest)c).setSeed(fold);
                
                break;
            case "ContractRotationForest1Day":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setHourLimit(24);
                ((ContractRotationForest)c).setSeed(fold);
                break;
            case "ContractRotationForest5Minutes":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setMinuteLimit(5);
                ((ContractRotationForest)c).setSeed(fold);
                break;
            case "ContractRotationForest30Minutes":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setMinuteLimit(30);
                ((ContractRotationForest)c).setSeed(fold);
                break;
            case "ContractRotationForest1Hour":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setHourLimit(1);
                ((ContractRotationForest)c).setSeed(fold);
                break;
            case "ContractRotationForest2Hour":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setHourLimit(2);
                break;
            case "ContractRotationForest3Hour":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setHourLimit(3);
                ((ContractRotationForest)c).setSeed(fold);
                break;
            case "ContractRotationForest12Hour":
                c= new ContractRotationForest();
                ((ContractRotationForest)c).setHourLimit(12);
                ((ContractRotationForest)c).setSeed(fold);
                break;
            
            case "ShapeletI": case "Shapelet_I": case "ShapeletD": case "Shapelet_D": case  "Shapelet_Indep"://Multivariate version 1
                c=new MultivariateShapeletTransformClassifier();
//Default to 1 day max run: could do this better
                ((MultivariateShapeletTransformClassifier)c).setOneDayLimit();
                ((MultivariateShapeletTransformClassifier)c).setSeed(fold);
                ((MultivariateShapeletTransformClassifier)c).setTransformType(classifier);
                break;
            case "ED_I":
                c=new NN_ED_I();
                break;
            case "DTW_I":
                c=new NN_DTW_I();
                break;
            case "DTW_D":
                c=new NN_DTW_D();
                break;
            case "DTW_A":
                c=new NN_DTW_A();
                break;
//TIME DOMAIN CLASSIFIERS   
            
            case "ED":
                c=new ED1NN();
                break;
            case "C45":
                c=new J48();
                break;
            case "NB":
                c=new NaiveBayes();
                break;
            case "SVML":
                c=new SMO();
                PolyKernel p=new PolyKernel();
                p.setExponent(1);
                ((SMO)c).setKernel(p);
                ((SMO)c).setRandomSeed(fold);
                ((SMO)c).setBuildLogisticModels(true);
                break;
            case "SVMQ": case "SVMQuad":
                c=new SMO();
                PolyKernel poly=new PolyKernel();
                poly.setExponent(2);
                ((SMO)c).setKernel(poly);
                ((SMO)c).setRandomSeed(fold);
                ((SMO)c).setBuildLogisticModels(true);

                
                /*                svm=new TunedSVM();
                svm.setKernelType(TunedSVM.KernelType.QUADRATIC);
                svm.optimiseParas(false);
                svm.optimiseKernel(false);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                c= svm;
 */               break;
            case "SVMRBF": 
                c=new SMO();
                RBFKernel rbf=new RBFKernel();
                rbf.setGamma(0.5);
                ((SMO)c).setC(5);
                ((SMO)c).setKernel(rbf);
                ((SMO)c).setRandomSeed(fold);
                ((SMO)c).setBuildLogisticModels(true);
                
                break;
            case "BN":
                c=new BayesNet();
                break;
            case "MLP":
                c=new MultilayerPerceptron();
                break;
            case "BaggingREP":
                Bagging bag = new Bagging();
                bag.setClassifier(new REPTree());
                bag.setNumIterations(500);
                bag.setBagSizePercent(70);
                c = bag;
                break;
            case "LogitBoost":
                LogitBoost lb = new LogitBoost();
                lb.setNumFolds(2);
                lb.setNumIterations(300);
                c = lb;
                break;
            case "LogitBoostEpsilon3":
                LogitBoost lb3 = new LogitBoost();
                lb3.setNumFolds(2);
                lb3.setNumIterations(300);
                lb3.setLikelihoodThreshold(1e-3);
                c = lb3;
                break;
            case "LogitBoostEpsilon6":
                LogitBoost lb6 = new LogitBoost();
                lb6.setNumFolds(2);
                lb6.setNumIterations(300);
                lb6.setLikelihoodThreshold(1e-6);
                c = lb6;
                break;
            case "LogitBoostEpsilon9":
                LogitBoost lb9 = new LogitBoost();
                lb9.setNumFolds(2);
                lb9.setNumIterations(300);
                lb9.setLikelihoodThreshold(1e-9);
                c = lb9;
                break;
            case "AdaBoostM1DS":
                //decision stump
                AdaBoostM1 ada = new AdaBoostM1();
                ada.setNumIterations(500);
                c = ada;
                break;

            case "Logistic":
                c= new Logistic();
                break;
            case "NN":
                kNN k=new kNN(100);
                k.setCrossValidate(true);
                k.normalise(false);
                k.setDistanceFunction(new EuclideanDistance());
                return k;
            case "HESCA":
            case "CAWPE":
                c=new CAWPE();
                ((CAWPE)c).setRandSeed(fold);
                break;
            case "CAWPEnoLogistic":
                c=new CAWPE();
                ((CAWPE)c).setRandSeed(fold);
                ((CAWPE)c).setDefaultCAWPESettings_NoLogistic();
                break;
            case "CAWPEPLUS":
                c=new CAWPE();
                ((CAWPE)c).setRandSeed(fold);                
                ((CAWPE)c).setAdvancedCAWPESettings();
                break;
            case "CAWPEFROMFILE":
                String[] classifiers={"XGBoost","RandF","RotF"};
                c=new CAWPE();
                ((CAWPE)c).setRandSeed(fold);  
                ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                ((CAWPE)c).setResultsFileLocationParameters(horribleGlobalPath, nastyGlobalDatasetName, fold);
                
                ((CAWPE)c).setClassifiersNamesForFileRead(classifiers);
                break;
            case "CAWPE_AS_COTE":
                String[] cls={"TSF","ST","SLOWDTWCV","BOSS"};
                c=new CAWPE();
                ((CAWPE)c).setRandSeed(fold);  
                ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                ((CAWPE)c).setResultsFileLocationParameters(horribleGlobalPath, nastyGlobalDatasetName, fold);
                ((CAWPE)c).setClassifiersNamesForFileRead(cls);
                break;

//ELASTIC CLASSIFIERS     
            case "EE": case "ElasticEnsemble":
                c=new ElasticEnsemble();
                break;
            case "DTW":
                c=new DTW1NN();
                ((DTW1NN )c).setWindow(1);
                break;
            case "SLOWDTWCV":
//                c=new DTW1NN();
                c=new SlowDTW_1NN();
                ((SlowDTW_1NN)c).optimiseWindow(true);
                break;
            case "DTWCV":
//                c=new DTW1NN();
//                c=new FastDTW_1NN();
//                ((FastDTW_1NN)c).optimiseWindow(true);
//                break;
//            case "FastDTWWrapper":
                c= new FastDTWWrapper();
                break;
            case "DD_DTW":
                c=new DD_DTW();
                break;
            case "DTD_C":
                c=new DTD_C();
                break;
            case "CID_DTW":
                c=new NN_CID();
                ((NN_CID)c).useDTW();
                break;
            case "MSM":
                c=new MSM1NN();
                break;
            case "TWE":
                c=new MSM1NN();
                break;
            case "WDTW":    
                c=new WDTW1NN();
                break;
                
            case "LearnShapelets": case "LS":
                c=new LearnShapelets();
                break;
            case "FastShapelets": case "FS":
                c=new FastShapelets();
                break;
            case "ShapeletTransform": case "ST": case "ST_Ensemble": case "ShapeletTransformClassifier":
                c=new ShapeletTransformClassifier();
//Default to 1 day max run: could do this better
                ((ShapeletTransformClassifier)c).setOneDayLimit();
                ((ShapeletTransformClassifier)c).setSeed(fold);
                break;
            case "RISE":
                c=new RISE();
                break;
            case "TSBF":
                c=new TSBF();
                break;
            case "BOP": case "BoP": case "BagOfPatterns":
                c=new BagOfPatterns();
                break;
            case "BOSS": case "BOSSEnsemble": 
                c=new BOSS();
                break;
            case "WEASEL":
                c = new WEASEL();
                ((WEASEL)c).setSeed(fold);
                break;
            case "TSF":
                c=new TSF();
                break;
             case "SAXVSM": case "SAX": 
                c=new SAXVSM();
                break;
             case "LPS":
                c=new LPS();
                break; 
             case "FlatCOTE":
                c=new FlatCote();
                break; 
             case "HiveCOTE": case "HIVECOTE": case "HiveCote": case "Hive-COTE":
                c=new HiveCote();
                ((HiveCote)c).setContract(24);
                break; 
              
            
           default:
                System.out.println("UNKNOWN CLASSIFIER "+classifier);
                System.exit(0);
//                throw new Exception("Unknown classifier "+classifier);
        }
        return c;
    }
       
//Threaded version
    public void run(){
        try {      
            System.out.print("Running ");
            for(String str:args)
                System.out.print(str+" ");
            System.out.print("\n");
            singleExperiment(args);
//            Experiments.singleClassifierAndFoldTrainTestSplit(args);
        } catch (Exception ex) {
            System.out.println("ERROR, cannot run experiment :");
            for(String str:args)
                System.out.print(str+",");
        }
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
    public static void sanityCheck(String c) throws Exception{
        experiments.DataSets.problemPath="\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\TSCProblems";
        experiments.DataSets.resultsPath="C:\\Temp\\";
        threadClassifier=c;
        depreciatedThreadedExperiment("TSCProblems",1,10);

    }
    public static void debugExperiment() throws Exception{
//        sanityCheck("DTWCV");
//        System.exit(0);
    //Debug run
        experiments.DataSets.problemPath="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Data/TSCProblems/";
        experiments.DataSets.resultsPath="C:\\Temp\\";
        File f=new File(experiments.DataSets.resultsPath);
        if(!f.isDirectory()){
            f.mkdirs();
        }
        generateTrainFiles=false;
        checkpoint=false;
        parameterNum=0;
        debug=true;
            String[] newArgs={"FastDTWWrapper","ItalyPowerDemand","1"};
            Experiments.singleClassifierAndFoldTrainTestSplit(newArgs);
//                for(String str:DataSets.UCIContinuousFileNames){
//                }
        System.exit(0);

        
    }
    public static String horribleGlobalPath="";
    public static String nastyGlobalDatasetName="";
    public static void main(String[] args) throws Exception{

//IF args are passed, it is a cluster run. Otherwise it is a local run, either threaded or not
        debug=false;

//        foldsInFile=true;
        for(String str:args)
            System.out.println(str);
        if(args.length<6){
            boolean threaded=false;
            if(debug){
                debugExperiment();
            }
            else{ 
    //Args 1 and 2 are problem and results path
                String[] newArgs=new String[6];
                newArgs[0]="//cmptscsvr.cmp.uea.ac.uk/ueatsc/BagsSDM/Data/";//All on the beast now
                newArgs[1]="//cmptscsvr.cmp.uea.ac.uk/ueatsc/BagsSDM/Results/";
    //Arg 3 argument is whether to cross validate or not and produce train files
                newArgs[2]="true";
    // Arg 4,5,6 Classifier, Problem, Fold  
                String[] names={"CAWPE_AS_COTE"};
                
                for(String str:names){
                    newArgs[3]=str;
//These are set in the localX method
//              newArgs[4]="Adiac";
//                newArgs[5]="1";
//                String[] problems=DataSets.tscProblems85;
                    String[] problems=new String[]{"psudo2BagsTwoClassHisto"};
                    nastyGlobalDatasetName=problems[0];
                            
                    //"GTtoSieveTwoClassHisto","SieveBagsTwoClassHisto",
                        //"FakeBagsTwoClassHisto","FakeBagsFiveClassHisto"};
                //"BagsTwoClassHisto","BagsFiveClassHisto", "leaveOutOneElectricalItemHisto",
//                "GTtoSieveTwoClassHisto","leaveOutOneElectricalItemHisto","SieveBagsTwoClassHisto"};
                    int folds=45;
                    threaded=false;
                    horribleGlobalPath="\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\BagsSDM\\Results\\";
                    if(threaded){//Do problems listed threaded 
                        localThreadedRun(newArgs,problems,folds);
                    }
                    else //Do the problems listed sequentially
                        localSequentialRun(newArgs,problems,folds);
                }
            }


//Threaded run
 //             threadedExperiment("cmpv2202398");  
 /*           generateTrainFiles=false;
            parameterNum=0;
            threadClassifier="FastDTWWrapper";
            threadedExperiment("ajb17pc",1,1);  
//             threadedExperiment("cmpv2264419");  

// //             threadedExperiment("UCIContinuous");  
//              threadedExperiment("LargeProblems");  
            }
  */          
            
        }
        else{    
            singleExperiment(args);
        }
    
    }
    public static void depreciatedThreadedExperiment(String dataSet, int startFold, int endFold) throws Exception{
        
        int cores = Runtime.getRuntime().availableProcessors();        
        System.out.println("# cores ="+cores);
 //     cores=1; //debug       
        
        ExecutorService executor = Executors.newFixedThreadPool(cores);
        Experiments exp;
        experiments.DataSets.problemPath="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Data/"+dataSet+"/";//Problem Path
        DataSets.resultsPath="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Results/"+dataSet+"/";         //Results path
        String[] problems= experiments.DataSets.tscProblems85;
        parameterNum=0;   
        String classifier=threadClassifier;
        switch(dataSet){
            case "UCIContinuous"://Do all UCI
                problems= experiments.DataSets.UCIContinuousFileNames;
            break;
            case "TSCProblems": case "TSC Problems": //Do all Repo
                problems= experiments.DataSets.tscProblems85;
            break;
            
            
        }
        if(dataSet.equals("cmpv2202398")){
            experiments.DataSets.problemPath="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Data/UCIContinuous/";//Problem Path
            experiments.DataSets.resultsPath="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Results/UCIContinuous/";         //Results path
            problems=cmpv2202398;
            
        }
        else if(dataSet.equals("cmpv2264419")){
            experiments.DataSets.problemPath="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Data/UCIContinuous/";//Problem Path
            experiments.DataSets.resultsPath="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Results/UCIContinuous/";         //Results path
            problems=cmpv2264419;
            
        }
        else if(dataSet.equals("ajb17pc")){
            experiments.DataSets.problemPath="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Data/UCIContinuous/";//Problem Path
            experiments.DataSets.resultsPath="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Results/UCIContinuous/";         //Results path
            problems=ajb17pc;
            
        }
            
        ArrayList<String> names=new ArrayList<>();
        for(String str:problems)
            names.add(str);
//        Collections.reverse(names);
        generateTrainFiles=true;
        checkpoint=true;
        for(int i=0;i<names.size();i++){//Iterate over problems
//            if(isPig(names.get(i)))
//                continue;
            for(int j=startFold;j<=endFold;j++){//Iterate over folds
                String[] args=new String[3];
                args[0]=classifier;
                args[1]=names.get(i);
                args[2]=""+j;
                exp=new Experiments();
                exp.args=args;
                executor.execute(exp);
            }
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
        System.out.println("Finished all threads");            
        
    }
    public static boolean isPig(String str){
//        if(str.equals("adult")||str.equals("miniboone")||str.equals("chess-krvk"))
//            return true;
        return false;
        
    }
    public static void singleExperiment(String[] args) throws Exception{
            experiments.DataSets.problemPath=args[0];
            experiments.DataSets.resultsPath=args[1];
//Arg 3 argument is whether to cross validate or not and produce train files
            generateTrainFiles=Boolean.parseBoolean(args[2]);
            File f=new File(experiments.DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
                f.setWritable(true, false);
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
            if(args.length>=8){
                parameterNum=Integer.parseInt(args[7]);
            }
            Experiments.singleClassifierAndFoldTrainTestSplit(newArgs);        
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
   
        String predictions = experiments.DataSets.resultsPath+classifier+"/Predictions/"+problem;
        File f=new File(predictions);
        if(!f.exists())
            f.mkdirs();
        
        //Check whether fold already exists, if so, dont do it, just quit
        if(!experiments.CollateResults.validateSingleFoldFile(predictions+"/testFold"+fold+".csv"))
        {
            Classifier c=setClassifier(classifier,fold);
            Instances[] data;
            data = sampleDataset(problem, fold);
            

            if(parameterNum>0 && c instanceof ParameterSplittable)//Single parameter fold
            {
                checkpoint=false;
//Check if it already exists, if it does, exit
                if(experiments.CollateResults.validateSingleFoldFile(predictions+"/fold"+fold+"_"+parameterNum+".csv")){ //Exit
                    System.out.println("Fold "+predictions+"/fold"+fold+"_"+parameterNum+".csv  already exists");
                    return; //Aready done
                }
            }
            
            double acc = singleClassifierAndFoldTrainTestSplit(data[0],data[1],c,fold,predictions);
            System.out.println("Classifier="+classifier+", Problem="+problem+", Fold="+fold+", Test Acc,"+acc);
        }
    }
    
    public static Instances[] sampleDataset(String problem, int fold) throws Exception {
        Instances[] data = new Instances[2];
        
        File trainFile=new File(experiments.DataSets.problemPath+problem+"/"+problem+fold+"_TRAIN.arff");
        File testFile=new File(experiments.DataSets.problemPath+problem+"/"+problem+fold+"_TEST.arff");
        
        foldsInFile = foldsInFile || (trainFile.exists() && testFile.exists());
        
//Shapelet special case, hard coded,because all folds are pre-generated             
        if(foldsInFile){
            if(!trainFile.exists()||!testFile.exists())
                throw new Exception(" Problem files "+ experiments.DataSets.problemPath+problem+"/"+problem+fold+"_TRAIN.arff not found");
            data[0]=ClassifierTools.loadData(experiments.DataSets.problemPath+problem+"/"+problem+fold+"_TRAIN");
            data[1]=ClassifierTools.loadData(experiments.DataSets.problemPath+problem+"/"+problem+fold+"_TEST");
        }
        else{
//If there is a train test split, use that. Otherwise, randomly split by split proportion            
            trainFile=new File(experiments.DataSets.problemPath+problem+"/"+problem+"_TRAIN.arff");
            testFile=new File(experiments.DataSets.problemPath+problem+"/"+problem+"_TEST.arff");
            if(!trainFile.exists()||!testFile.exists())
                singleFile=true;
            if(singleFile){
                Instances all = ClassifierTools.loadData(experiments.DataSets.problemPath+problem+"/"+problem);
                if(all.checkForAttributeType(Attribute.RELATIONAL))
                    data = MultivariateInstanceTools.resampleMultivariateInstances(all, fold, SPLITPROP);            
                else
                    data = InstanceTools.resampleInstances(all, fold, SPLITPROP);            
                    
            }else{
                data[0]=ClassifierTools.loadData(trainFile.getAbsolutePath());
                data[1]=ClassifierTools.loadData(testFile.getAbsolutePath());
                if(data[0].checkForAttributeType(Attribute.RELATIONAL))
                    data = MultivariateInstanceTools.resampleMultivariateTrainAndTestInstances(data[0],data[1], fold);                            
                else
                    data=InstanceTools.resampleTrainAndTestInstances(data[0], data[1], fold);
            }
        }
        return data;
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
//If TunedRandForest or TunedRotForest need to let the classifier know the number of attributes 
//n orderto set parameters
            checkpoint=false;
            ((ParameterSplittable)c).setParametersFromIndex(parameterNum);
//            System.out.println("classifier paras =");
            trainFoldPath="/fold"+fold+"_"+parameterNum+".csv";
            generateTrainFiles=true;
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
                if(c instanceof TrainAccuracyEstimate){ //Classifier will perform cv internally
                    ((TrainAccuracyEstimate)c).writeCVTrainToFile(resultsPath+trainFoldPath);
                    File f=new File(resultsPath+trainFoldPath);
                    if(f.exists())
                        f.setWritable(true, false);
                }
                else{ // Need to cross validate here
                    CrossValidator cv = new CrossValidator();
                    cv.setSeed(fold);
                    int numFolds = Math.min(train.numInstances(), numCVFolds);
                    cv.setNumFolds(numFolds);
                    trainResults=cv.crossValidateWithStats(c,train);
                }
            }
            
            //Build on the full train data here
//            long buildTime=System.nanoTime();
            long buildTime=System.currentTimeMillis();
            c.buildClassifier(train);
//            buildTime=System.nanoTime()-buildTime;
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
                    File f=new File(resultsPath+trainFoldPath);
                    if(f.exists())
                        f.setWritable(true, false);
                    
                }
            }
            if(parameterNum==0)//Not a single parameter fold
            {  
                //Start of testing, only doing this if the test file doesnt exist
                //This is checked before the buildClassifier also, but we have a special case for the file builder
                //that copies the results over in buildClassifier. No harm in checking again!
                if(!CollateResults.validateSingleFoldFile(resultsPath+testFoldPath)){
                    int numInsts = test.numInstances();
                    int pred;
                    testResults = new ClassifierResults(test.numClasses());
                    double[] trueClassValues = test.attributeToDoubleArray(test.classIndex()); //store class values here
                        
//                    long testTime = System.nanoTime();
                    for(int testInstIndex = 0; testInstIndex < numInsts; testInstIndex++) {
                        test.instance(testInstIndex).setClassMissing();//and remove from each instance given to the classifier (just to be sure)

                        //make prediction
                        double[] probs=c.distributionForInstance(test.instance(testInstIndex));
                        testResults.storeSingleResult(probs);
                    }
//                    testTime=System.nanoTime()-testTime;
                    testResults.finaliseResults(trueClassValues); 

                    //Write results
                    OutFile testOut=new OutFile(resultsPath+testFoldPath);
                    testOut.writeLine(test.relationName()+","+c.getClass().getName()+",test");
                    
                    //START  JAMES L FOR HESCA TEST TIMES
                    if(c instanceof SaveParameterInfo) {
                      testOut.writeLine(((SaveParameterInfo)c).getParameters());
                    }
                    else
                        testOut.writeLine("No parameter info");
//                    testOut.writeLine("BuildTime,"+testTime+",No parameter info");
                    //END    JAMES L FOR HESCA TEST TIMES

                    testOut.writeLine(testResults.acc+"");
                    testOut.writeString(testResults.writeInstancePredictions());
                    testOut.closeFile();
                    File f=new File(resultsPath+testFoldPath);
                    if(f.exists())
                        f.setWritable(true, false);
                    
                }
                return testResults.acc;
            }
            else
                 return 0;//trainResults.acc;   
        } catch(Exception e) {
            System.out.println(" Error ="+e+" in method simpleExperiment");
            e.printStackTrace();
            System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
            System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes and "+test.numInstances()+" instances");
            System.out.println(" Classifier ="+c.getClass().getName()+" fold = "+fold);
            System.out.println(" Results path is "+ resultsPath);
                    
            return Double.NaN;
        }
    }    

    
    public static void localSequentialRun(String[] standardArgs,String[] problemList, int folds) throws Exception{
        for(String str:problemList){
            System.out.println("Problem ="+str);
            nastyGlobalDatasetName=str;
            for(int i=1;i<=folds;i++){
                standardArgs[4]=str;
                standardArgs[5]=i+"";
                singleExperiment(standardArgs);
                
            }
        }
        
        
    }
    public static void localThreadedRun(String[] standardArgs,String[] problemList, int folds) throws Exception{
        int cores = Runtime.getRuntime().availableProcessors();        
        System.out.println("# cores ="+cores);
        cores=cores/2;
        System.out.println("# threads ="+cores);
        ExecutorService executor = Executors.newFixedThreadPool(cores);
        Experiments exp;
        for(String str:problemList){
            nastyGlobalDatasetName=str;
            for(int i=1;i<=folds;i++){
                String[] args=new String[standardArgs.length];//Need to clone them!
                for(int j=0;j<standardArgs.length;j++)
                    args[j]=standardArgs[j];
                args[4]=str;
                args[5]=i+"";
                exp=new Experiments();
                exp.args=args;
                executor.execute(exp);
            }
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
        System.out.println("Finished all threads");            
    }
    
   public static void tonyTest() throws Exception{
        Instances all = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\UCI Problems\\hayes-roth\\hayes-roth");
        Instances[] data = InstanceTools.resampleInstances(all,0, .5);            
        RandomForest rf=new RandomForest();
        rf.setMaxDepth(0);
        rf.setNumFeatures(1);
        rf.setNumTrees(10);
        CrossValidator cv = new CrossValidator();
        ClassifierResults tempResults=cv.crossValidateWithStats(rf,data[0]);
                    tempResults.setName("RandFPara");
                    tempResults.setParas("maxDepth,"+rf.getMaxDepth()+",numFeatures,"+rf.getNumFeatures()+",numTrees,"+rf.getNumTrees());
                    System.out.println(tempResults.writeResultsFileToString());


}
 
}






