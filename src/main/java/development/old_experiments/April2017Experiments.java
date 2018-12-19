/**
 *
 * @author ajb
 *local class to run experiments with the UCI data


*/
package development.old_experiments;

import development.DataSets;
import development.RotationForestLimitedAttributes;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.Classifier;
import vector_classifiers.TunedSVM;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.RotationForest;
import vector_classifiers.TunedRotationForest;
import vector_classifiers.CAWPE;
import timeseriesweka.classifiers.ensembles.SaveableEnsemble;
import weka.classifiers.trees.RandomForest;
import vector_classifiers.TunedRandomForest;
import weka.core.Instances;


public class April2017Experiments{
    public static String[] classifiers={"SVM"};
    public static double propInTrain=0.5;
    public static int folds=100;
    public static Random rand=new Random(0);
    static String[] UCIContinuousFileNames={"abalone","acute-inflammation","acute-nephritis","adult","annealing","arrhythmia","audiology-std","balance-scale","balloons","bank","blood","breast-cancer","breast-cancer-wisc","breast-cancer-wisc-diag","breast-cancer-wisc-prog","breast-tissue","car","cardiotocography-10clases","cardiotocography-3clases",
        "chess-krvk","chess-krvkp","congressional-voting","conn-bench-sonar-mines-rocks","conn-bench-vowel-deterding",
        "connect-4","contrac","credit-approval","cylinder-bands","dermatology","echocardiogram","ecoli","energy-y1","energy-y2","fertility","flags","glass","haberman-survival","hayes-roth","heart-cleveland","heart-hungarian","heart-switzerland","heart-va","hepatitis","hill-valley","horse-colic","ilpd-indian-liver","image-segmentation","ionosphere","iris","led-display","lenses","letter","libras","low-res-spect","lung-cancer","lymphography","magic","mammographic",
        "miniboone","molec-biol-promoter","molec-biol-splice","monks-1","monks-2","monks-3","mushroom","musk-1","musk-2","nursery","oocytes_merluccius_nucleus_4d","oocytes_merluccius_states_2f","oocytes_trisopterus_nucleus_2f","oocytes_trisopterus_states_5b","optical","ozone","page-blocks","parkinsons","pendigits","pima","pittsburg-bridges-MATERIAL","pittsburg-bridges-REL-L","pittsburg-bridges-SPAN","pittsburg-bridges-T-OR-D","pittsburg-bridges-TYPE","planning","plant-margin","plant-shape","plant-texture","post-operative","primary-tumor","ringnorm","seeds","semeion","soybean","spambase","spect","spectf","statlog-australian-credit","statlog-german-credit","statlog-heart","statlog-image","statlog-landsat","statlog-shuttle","statlog-vehicle","steel-plates","synthetic-control","teaching","thyroid","tic-tac-toe","titanic","trains","twonorm","vertebral-column-2clases","vertebral-column-3clases","wall-following","waveform","waveform-noise","wine","wine-quality-red","wine-quality-white","yeast","zoo"};
    static boolean debug=true;
    
    static String[] files=UCIContinuousFileNames;
//Parameter ranges for search, use same for C and gamma   
    static double[] svmParas={0.00390625, 0.015625, 0.0625, 0.25, 0.5, 1, 2, 4, 16, 256};
//Parameter ranges for trees for randF and rotF
    static int[] numTrees={10,50,100,200,300,400,500,600,700,800,900,1000,1250,1500,1750,2000};

    public static void generateAllRepoFolds(String source,String dest) throws IOException{
        for(String problem:DataSets.tscProblems85){
            File f=new File(source+problem+"/"+problem+"_TRAIN.arff"); 
            File f2=new File(source+problem+"/"+problem+"_TEST.arff"); 
            if(f.exists()&&f2.exists()){
                Instances train=ClassifierTools.loadData(f);
                Instances test=ClassifierTools.loadData(f2);
                 for(int i=0;i<folds;i++){
                    Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, i);
                    File of=new File(dest+problem);
                    if(!of.isDirectory())
                        of.mkdir();
                    OutFile outTrain=new OutFile(dest+problem+"/"+problem+i+"_TRAIN.arff");
                    OutFile outTest=new OutFile(dest+problem+"/"+problem+i+"_TEST.arff");
                    outTrain.writeString(data[0].toString());
                    outTest.writeString(data[1].toString());
                    
                }
             }
            else
                System.out.println("MISSING "+source+problem+"/"+problem+"_TRAIN.arff");
        }
        
    }
    public static void generateAllUCIFolds(String source, String dest) throws IOException{
        for(String problem:UCIContinuousFileNames){
//Load full file
            File f=new File(source+problem+"/"+problem+".arff"); 
            if(f.exists()){
                Instances data=ClassifierTools.loadData(f);
    //
                for(int i=0;i<folds;i++){
                    Instances[]  split=InstanceTools.resampleInstances(data, i, propInTrain);
                    File of=new File(dest+problem);
                    if(!of.isDirectory())
                        of.mkdir();
                    OutFile outTrain=new OutFile(dest+problem+"/"+problem+i+"_TRAIN.arff");
                    OutFile outTest=new OutFile(dest+problem+"/"+problem+i+"_TEST.arff");
                    outTrain.writeString(split[0].toString());
                    outTest.writeString(split[1].toString());
                }
            }
            else
                System.out.println("MISSING "+source+problem+"/"+problem+".arff");
        }
        
    }

    
    
    public static void generateTestTRainUCIFolds(String source, String dest) throws IOException{
        for(String problem:UCIContinuousFileNames){
//Load full file
            File f=new File(source+problem+"/"+problem+".arff"); 
            if(f.exists()){
                Instances data=ClassifierTools.loadData(f);
                Instances[]  split=InstanceTools.resampleInstances(data, 0, propInTrain);
                File of=new File(dest+problem);
                if(!of.isDirectory())
                    of.mkdir();
                OutFile outTrain=new OutFile(dest+problem+"/"+problem+"_TRAIN.arff");
                OutFile outTest=new OutFile(dest+problem+"/"+problem+"_TEST.arff");
                outTrain.writeString(split[0].toString());
                outTest.writeString(split[1].toString());
            }
            else
                System.out.println("MISSING "+source+problem+"/"+problem+".arff");
        }
        
    }
    
    public static void timingNormalisation(String file) throws Exception{
        OutFile out=new OutFile(file);
        for(int i=0;i<10;i++){
            Instances train=ClassifierTools.loadData(DataSets.problemPath+"Yoga/Yoga");
            RotationForest rf=new RotationForest();
            long t1=System.currentTimeMillis();
            rf.buildClassifier(train);
            long t2=System.currentTimeMillis();
            out.writeLine(i+","+(t2-t1));
            System.out.println("Run "+i+" Time ="+(t2-t1)+" milliseconds ");
            rf=null;
            System.gc();
        }
            
        
    }
    
public static boolean deleteDirectory(File directory) {
    if(directory.exists()){
        File[] files = directory.listFiles();
        if(null!=files){
            for(int i=0; i<files.length; i++) {
                if(files[i].isDirectory())
                    deleteDirectory(files[i]);
                else
                    files[i].delete();
            }
        }
    }
    return(directory.delete());
}    
    
    public static void generateScripts(boolean grace,int mem, String jar,String[] fileNames, String dir){
//Generates cluster scripts for allTest combos of classifier and data set
//Generates txt files to run jobs for a single classifier        
        String path=DataSets.dropboxPath+"Code\\Cluster Scripts\\"+dir+"\\";
        File f=new File(path);
        deleteDirectory(f);
        f.delete();
        f.mkdirs();
        ArrayList<String> list=new ArrayList<>();
        for(String s:classifiers){
            OutFile of2;
            if(grace)
                of2=new OutFile(path+s+"Grace.txt");
            else
                of2=new OutFile(path+s+".txt");
            for(String a:fileNames){
                OutFile of;
                if(grace)
                    of = new OutFile(path+s+a+"Grace.bsub");
                else
                    of = new OutFile(path+s+a+".bsub");
                of.writeLine("#!/bin/csh");
                if(grace)
                    of.writeLine("#BSUB -q short");
                else
                    of.writeLine("#BSUB -q long-eth");
                of.writeLine("#BSUB -J "+s+a+"[1-"+folds+"]");
                of.writeLine("#BSUB -oo output/"+a+".out");
                of.writeLine("#BSUB -eo error/"+a+".err");
                if(grace){
                    of.writeLine("#BSUB -R \"rusage[mem="+mem+"]\"");
                    of.writeLine("#BSUB -M "+mem);
                    of.writeLine(" module add java/jdk/1.8.0_31");
                }
                else{
                    of.writeLine("#BSUB -R \"rusage[mem="+(2000+mem)+"]\"");
                    of.writeLine("#BSUB -M "+(2000+mem));
                    of.writeLine("module add java/jdk1.8.0_51");
                }
                of.writeLine("java -jar "+jar+".jar "+s+" "+a+" $LSB_JOBINDEX");                
                if(grace)
                    of2.writeLine("bsub < Scripts/"+dir+"/"+s+a+"Grace.bsub");
                else
                    list.add("bsub < Scripts/"+dir+"/"+s+a+".bsub");
            }
            if(!grace){
                Collections.reverse(list);
                for(String str:list)
                    of2.writeLine(str);
            }
        }
    } 
    public static boolean foldComplete(String path, int fold, int numTrain,int numTest){
//Check both train and test present
      File f=new File(path+"//testFold"+fold+".csv");
      File f2=new File(path+"//trainFold"+fold+".csv");
      if(!f.exists()||!f2.exists())//Neither exist
          return false;
      else{
          InFile inf1=new InFile(path+"//testFold"+fold+".csv");
          InFile inf2=new InFile(path+"//testFold"+fold+".csv");
//Check number of lines
          int c1=inf1.countLines();
          int c2=inf2.countLines();
          if(c1<(3) || c2<(3))
              return false;
      }
      return true;
    }
  /**
     * collates the differences between test and train into a single file
     * @param folds
     * @param cls
     */
    public static void collateTrain(){
        String base="C:\\Research\\Papers\\2017\\ECML Standard Parameters\\Section 4 Bakeoff\\TuneCompare\\";
        OutFile test=new OutFile(base+"inOneLineTest.csv");
        OutFile train=new OutFile(base+"inOneLineTrain.csv");
        InFile svmTrain=new InFile(base+"TunedSVMTrainCV.csv");
        InFile svmTest=new InFile(base+"TunedSVMTest.csv");
        InFile randFTrain=new InFile(base+"TunedRandFTrainCV.csv");
        InFile randFTest=new InFile(base+"TunedRandFTest.csv");
        InFile rotFTrain=new InFile(base+"TunedRotFTrainCV.csv");
        InFile rotFTest=new InFile(base+"TunedRotFTest.csv");
        for(String str:files){
            String[] svmTr=svmTrain.readLine().split(",");
            String[] svmTe=svmTest.readLine().split(",");
            String[] randFTr=randFTrain.readLine().split(",");
            String[] randFTe=randFTest.readLine().split(",");
            String[] rotFTr=rotFTrain.readLine().split(",");
            String[] rotFTe=rotFTest.readLine().split(",");
            int l=31;
            if(svmTr.length==l&& svmTe.length==l &&
                randFTr.length==l&& randFTe.length==l &&
                    rotFTr.length==l&& rotFTe.length==l
                    ){//OK, GOT THEM ALL
                for(int i=1;i<l;i++){
                    train.writeLine(svmTr[i]+","+randFTr[i]+","+rotFTr[i]);
                    test.writeLine(svmTe[i]+","+randFTe[i]+","+rotFTe[i]);
                }
            }
        }
    }


    public static void collateResults(int folds, boolean onCluster, String[] classif){
        if(onCluster)
           DataSets.resultsPath=DataSets.clusterPath+classif[0];

        String basePath=DataSets.resultsPath;
//1. Collate single folds into single classifier_problem files        
        for(int i=1;i<classif.length;i++){
            String cls=classif[i];
//Check classifier directory exists. 
            File f=new File(basePath+cls);
            if(f.isDirectory()){
//Write collated results for this classifier to a single file                
                OutFile clsResults=new OutFile(basePath+cls+"//"+cls+"Test.csv");
                OutFile trainResults=new OutFile(basePath+cls+"//"+cls+"TrainCV.csv");
                OutFile cPara=new OutFile(basePath+cls+"//"+cls+"ParameterC.csv");
                OutFile gammaPara=new OutFile(basePath+cls+"//"+cls+"ParameterGamma.csv");
                OutFile missing=null;
                int missingCount=0;
                for(String name:files){            
                    clsResults.writeString(name+",");
                    trainResults.writeString(name+",");
                    cPara.writeString(name+",");
                    gammaPara.writeString(name+",");
                    String path=basePath+cls+"//Predictions//"+name;
                    if(missing!=null && missingCount>0)
                        missing.writeString("\n");
                    missingCount=0;
                    for(int j=0;j<folds;j++){
    //Check fold exists
                        f=new File(path+"//testFold"+j+".csv");

                        if(f.exists() && f.length()>0){//This could fail if file only has partial probabilities on the line
 //This could fail if file only has partial probabilities on the line
    //Read in test ccuracy and store                    
    //Check fold exists
    //Read in test ccuracy and store
                            InFile inf=null;
                            String[] trainRes=null;
                            try{
                                inf=new InFile(path+"//testFold"+j+".csv");
                                inf.readLine();
                                trainRes=inf.readLine().split(",");//Stores train CV and parameter info
                                clsResults.writeString(inf.readDouble()+",");
                                if(trainRes.length>1){//There IS parameter info
                                    trainResults.writeString(Double.parseDouble(trainRes[1])+",");
                                    cPara.writeString(Double.parseDouble(trainRes[3])+",");
                                    if(trainRes.length>4)
                                        gammaPara.writeString(Double.parseDouble(trainRes[5])+",");
                                    else
                                        gammaPara.writeString(",");//Lazy!
                                }
                                else{
                                    trainResults.writeString(",");//Lazy!
                                    cPara.writeString(",");//Lazy!
                                    gammaPara.writeString(",");//Lazy!
                                    
                                }
                            }catch(Exception e){
                                System.out.println(" Error "+e+" in "+path);
                                trainResults.writeString(",");//Lazy!
                                cPara.writeString(",");//Lazy!
                                gammaPara.writeString(",");//Lazy!
                                if(trainRes!=null){
                                    System.out.println(" second line read has "+trainRes.length+" entries :");
                                    for(String str:trainRes)
                                        System.out.println(str);
                                }
//                                System.exit(1);
                            }finally{
                                if(inf!=null)
                                    inf.closeFile();

                            }
                        }
                        else{
                            if(missing==null)
                                missing=new OutFile(basePath+cls+"//"+cls+"MISSING.csv");
                            if(missingCount==0)
                                missing.writeString(name);
                            missingCount++;
                           missing.writeString(","+j);
                        }
                       
                    }
                    for(int j=0;j<folds;j++){
                    }
                    clsResults.writeString("\n");
                    trainResults.writeString("\n");
                    cPara.writeString("\n");
                    gammaPara.writeString("\n");
                }
                clsResults.closeFile();
                trainResults.closeFile();
                cPara.closeFile();
                gammaPara.closeFile();
            }
        }
//3. Merge classifier files into a single file with average accuracies
        //NEED TO REWRITE FOR TRAIN TEST DIFF
        OutFile acc=new OutFile(basePath+"CombinedAcc.csv");
        OutFile count=new OutFile(basePath+"CombinedCount.csv");
        for(int i=1;i<classif.length;i++){
            String cls=classif[i];
            acc.writeString(","+cls);
            count.writeString(","+cls);
        }
        acc.writeString("\n");
        count.writeString("\n");
        InFile[] allTest=new InFile[classif.length-1];
        for(int i=0;i<allTest.length;i++){
            String p=basePath+classif[i+1]+"/"+classif[i+1]+"Test.csv";
            if(new File(p).exists())
                allTest[i]=new InFile(p);
            else
                allTest[i]=null;//superfluous
        }
        for(int i=0;i<files.length;i++){
            acc.writeString(files[i]+",");
            count.writeString(files[i]+",");
            String prev="First";
            for(int j=0;j<allTest.length;j++){
                if(allTest[j]==null){
                    acc.writeString(",");
                    count.writeString("0,");
                }
                else{//Find mean
                    try{
                        String r=allTest[j].readLine();
                        String[] res=r.split(",");
                        count.writeString((res.length-1)+",");
                        double mean=0;
                        for(int k=1;k<res.length;k++){
                            mean+=Double.parseDouble(res[k]);
                        }
                        if(res.length>1){
                            acc.writeString((mean/(res.length-1))+",");
                        }
                        else{
                            acc.writeString(",");
                        }
                        prev=r;
                    }catch(Exception ex){
                        System.out.println("failed to read line: "+ex+" previous line = "+prev);
                    }
                }
            }
            acc.writeString("\n");
            count.writeString("\n");
        }
        for(InFile  inf:allTest)
            if(inf!=null)
                inf.closeFile();
        acc.closeFile();
        count.closeFile();
        
    }

    public static void collateTrainTestResults(int folds){
        String basePath="C:\\Research\\Results\\UCIResults\\";
//1. Collate single folds into single classifier_problem files        
        for(String cls:classifiers){
//Check classifier directory exists. 
            File f=new File(basePath+cls);
            if(f.isDirectory()){
//Write collated results for this classifier to a single file                
                OutFile clsResults=new OutFile(basePath+cls+"//"+cls+"TrainTestDiffs.csv");
                OutFile missing=null;
                int missingCount=0;
                for(int i=0;i<files.length;i++){
                    String name=files[i];
                    clsResults.writeString(files[i]+",");
                    String path=basePath+cls+"//Predictions//"+files[i];
                    if(missing!=null && missingCount>0)
                        missing.writeString("\n");
                    missingCount=0;
                    for(int j=0;j<folds;j++){
    //Check fold exists
                        f=new File(path+"//testFold"+j+".csv");
                        File f2=new File(path+"//trainFold"+j+".csv");

                        if((f2.exists() && f2.length()>0)&&(f.exists() && f.length()>0)){//This could fail if file only has partial probabilities on the line
 //This could fail if file only has partial probabilities on the line
    //Read in test ccuracy and store                    
    //Check fold exists
    //Read in test ccuracy and store                    
                            InFile inf=new InFile(path+"//testFold"+j+".csv");
                            inf.readLine();
                            inf.readLine();
                            double test=inf.readDouble();
                            inf=new InFile(path+"//trainFold"+j+".csv");
                            inf.readLine();
                            inf.readLine();
                            double train=inf.readDouble();
                            
                            clsResults.writeString((test-train)+",");    
                        }
                        else{
                            if(missing==null)
                                missing=new OutFile(basePath+cls+"//"+cls+"MISSING.csv");
                            if(missingCount==0)
                                missing.writeString(name);
                            missingCount++;
                           missing.writeString(","+j);
                        }
                    }
                    for(int j=0;j<folds;j++){
                    }
                    clsResults.writeString("\n");
                }
                clsResults.closeFile();
            }
        }
//3. Merge classifier files into a single file with average accuracies
        //NEED TO REWRITE FOR TRAIN TEST DIFF
        OutFile diff=new OutFile(basePath+"TrainTestDiff.csv");
        for(String cls:classifiers){
            diff.writeString(","+cls);
        }
        diff.writeString("\n");
        InFile[] allDiffs=new InFile[classifiers.length];
        for(int i=0;i<allDiffs.length;i++){
            String p=basePath+classifiers[i]+"//"+classifiers[i]+"TrainTestDiffs.csv";
            if(new File(p).exists())
                allDiffs[i]=new InFile(p);
            else
                allDiffs[i]=null;//superfluous
        }
        for(int i=0;i<files.length;i++){
            diff.writeString(files[i]+",");
            for(int j=0;j<allDiffs.length;j++){
                if(allDiffs[j]==null){
                    diff.writeString(",");
                }
                else{//Find mean
                    String[] res=allDiffs[j].readLine().split(",");
                    diff.writeString((res.length-1)+",");
                    double diffMean=0;
                    for(int k=1;k<res.length;k++){
                        diffMean+=Double.parseDouble(res[k]);
//                        diffMean+=Double.parseDouble(res[k])-Double.parseDouble(tr[k]);;
                    }
                    if(res.length>1){
                        diff.writeString((diffMean/(res.length-1))+",");
//                        diff.writeString((diffMean/(res.length-1))+",");
                    }
                    else{
                        diff.writeString(",");
//                        diff.writeString(",");
                    }
                }
            } 
            diff.writeString("\n");
            
        }
        
    }


    public static Classifier setClassifier(String classifier, int fold){
//RandF or RotF
        TunedRandomForest randF;
        TunedRotationForest r;
        switch(classifier){
            case "SVM":
                TunedSVM svm=new TunedSVM();
                svm.setKernelType(TunedSVM.KernelType.RBF);
                svm.optimiseParas(true);
                svm.optimiseKernel(false);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                return svm;
           case "RotF":
                r=new TunedRotationForest();
                r.setNumIterations(200);
                r.justBuildTheClassifier();
                return r;
            case "RandRotF1":
                RotationForestLimitedAttributes r3=new RotationForestLimitedAttributes();
                r3.setNumIterations(200);
                r3.setMaxNumAttributes(100);
                r3.justBuildTheClassifier();
                return r3;
            case "HESCA":
                String[] names={"RotFCV","RandFOOB","SVM"};
                Classifier[] c=new Classifier[3];
                c[0]=new IBk();
                c[1]=new IBk();
                c[2]=new IBk();
                CAWPE h = new CAWPE();
                h.setClassifiers(c, names, null);
                h.setDebug(true);
                
                
                return h;    
            case "RotFCV":
                r = new TunedRotationForest();
                r.setNumIterations(200);
                r.tuneParameters(false);
                r.estimateAccFromTrain(true);
                return r;
            case "RandFCV":
                randF = new TunedRandomForest();
                randF.tuneParameters(false);
                randF.setNumTrees(500);
                randF.debug(debug);
                randF.setSeed(fold);
                randF.setEstimateAcc(true);
                randF.setCrossValidate(true);
                return randF;
            case "RandFOOB":
                randF = new TunedRandomForest();
                randF.tuneParameters(false);
                randF.setNumTrees(500);
                randF.debug(debug);
                randF.setSeed(fold);
                randF.setEstimateAcc(true);
                randF.setCrossValidate(false);
                return randF; 

            default:
            throw new RuntimeException("Unknown classifier = "+classifier+" in Feb 2017 class");
        }
    }
    public static void singleClassifierAndFoldSingleDataSet(String[] args){
//first gives the problem file      
        String classifier=args[0];
        String problem=args[1];
        int fold=Integer.parseInt(args[2])-1;
   
        Classifier c=April2017Experiments.setClassifier(classifier,fold);
        Instances all=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem);
//        all.randomize(rand);
        
        Instances[] split=InstanceTools.resampleInstances(all, fold, propInTrain);
        File f=new File(DataSets.resultsPath+classifier);
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+classifier+"/Predictions"+"/"+problem;
        f=new File(predictions);
        if(!f.exists())
            f.mkdirs();
//Check whether fold already exists, if so, dont do it, just quit
        f=new File(predictions+"/testFold"+fold+".csv");
        if(!f.exists() || f.length()==0){
      //      of.writeString(problem+","); );
            if(c instanceof TrainAccuracyEstimate)
                ((TrainAccuracyEstimate)c).writeCVTrainToFile(predictions+"/trainFold"+fold+".csv");
            if(c instanceof CAWPE){
                System.out.println("Turning on file read ");
                  ((CAWPE)c).setResultsFileLocationParameters(DataSets.resultsPath, problem, fold);
                  ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
            }
            double acc =singleClassifierAndFoldSingleDataSet(split[0],split[1],c,fold,predictions);
            System.out.println(classifier+","+problem+","+fold+","+acc);
            
 //       of.writeString("\n");
        }
    }
    
    public static double singleClassifierAndFoldSingleDataSet(Instances train, Instances test, Classifier c, int fold,String resultsPath){
        double acc=0;
        int act;
        int pred;
// Save internal info for ensembles
//        if(c instanceof SaveableEnsemble)
//           ((SaveableEnsemble)c).saveResults(resultsPath+"/internalCV_"+fold+".csv",resultsPath+"/internalTestPreds_"+fold+".csv");
        OutFile p=null;
        try{              
            c.buildClassifier(train);
            StringBuilder str = new StringBuilder();
            DecimalFormat df=new DecimalFormat("##.######");
            for(int j=0;j<test.numInstances();j++)
            {
                act=(int)test.instance(j).classValue();

                test.instance(j).setClassMissing();//Just in case ....
                double[] probs=c.distributionForInstance(test.instance(j));
                pred=0;
                for(int i=1;i<probs.length;i++){
                    if(probs[i]>probs[pred])
                        pred=i;
                }
                if(act==pred)
                    acc++;
                str.append(act);
                str.append(",");
                str.append(pred);
                str.append(",");
                for(double d:probs){
                    str.append(",");
                    str.append(df.format(d));
                }
                str.append("\n");
            }
            acc/=test.numInstances();
           
            p=new OutFile(resultsPath+"/testFold"+fold+".csv");
            if(p==null) throw new Exception(" file wont open!! "+resultsPath+"/testFold"+fold+".csv");
            p.writeLine(train.relationName()+","+c.getClass().getName()+",test");
            if(c instanceof SaveParameterInfo){
              p.writeLine(((SaveParameterInfo)c).getParameters());
            }else
                p.writeLine("No parameter info");
            p.writeLine(acc+"");
            p.writeLine(str.toString());
        }catch(Exception e)
        {
                e.printStackTrace();
                System.out.println(" Error ="+e+" in method singleClassifierAndFold in class Feb2017");
                System.out.println(" Classifier = "+c.getClass().getName());
                System.out.println(" Results path="+resultsPath);
                System.out.println(" Outfile = "+p);
                System.out.println(" Train Split = "+train.toSummaryString());
                System.out.println(" Test Split = "+test.toSummaryString());
                e.printStackTrace();
                System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
                System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes"+test.numInstances()+" instances");

                System.exit(0);
        }
         return acc;
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
    public static void singleClassifierAndFoldTrainTestSplit(String[] args){
//first gives the problem file      
        String classifier=args[0];
        String problem=args[1];
        int fold=Integer.parseInt(args[2])-1;
   
        Classifier c=setClassifier(classifier,fold);
        Instances train=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TRAIN");
        Instances test=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TEST");
        File f=new File(DataSets.resultsPath+classifier);
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+classifier+"/Predictions";
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
        predictions=predictions+"/"+problem;
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
//Check whether fold already exists, if so, dont do it, just quit
        f=new File(predictions+"/testFold"+fold+".csv");
        if(!f.exists() || f.length()==0){
      //      of.writeString(problem+","); );
            if(c instanceof TrainAccuracyEstimate)
                ((TrainAccuracyEstimate)c).writeCVTrainToFile(predictions+"/trainFold"+fold+".csv");
            double acc =singleClassifierAndFoldTrainTestSplit(train,test,c,fold,predictions);
            System.out.println(classifier+","+problem+","+fold+","+acc);
            
 //       of.writeString("\n");
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
        Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, fold);
        double acc=0;
        int act;
        int pred;
// Save internal info for ensembles
        if(c instanceof SaveableEnsemble)
           ((SaveableEnsemble)c).saveResults(resultsPath+"/internalCV_"+fold+".csv",resultsPath+"/internalTestPreds_"+fold+".csv");
        try{              
            c.buildClassifier(data[0]);
            if(debug){
                if(c instanceof RandomForest)
                    System.out.println(" Number of features in MAIN="+((RandomForest)c).getNumFeatures());
            }
            StringBuilder str = new StringBuilder();
            DecimalFormat df=new DecimalFormat("##.######");
            for(int j=0;j<data[1].numInstances();j++)
            {
                act=(int)data[1].instance(j).classValue();
                data[1].instance(j).setClassMissing();//Just in case ....
                double[] probs=c.distributionForInstance(data[1].instance(j));
                pred=0;
                for(int i=1;i<probs.length;i++){
                    if(probs[i]>probs[pred])
                        pred=i;
                }
                if(act==pred)
                    acc++;
                str.append(act);
                str.append(",");
                str.append(pred);
                str.append(",,");
                for(double d:probs){
                    str.append(df.format(d));
                    str.append(",");
                }
                str.append("\n");
            }
            acc/=data[1].numInstances();
            OutFile p=new OutFile(resultsPath+"/testFold"+fold+".csv");
            p.writeLine(train.relationName()+","+c.getClass().getName()+",test");
            if(c instanceof SaveParameterInfo){
              p.writeLine(((SaveParameterInfo)c).getParameters());
            }else
                p.writeLine("No parameter info");
            p.writeLine(acc+"");
            p.writeLine(str.toString());
        }catch(Exception e)
        {
                System.out.println(" Error ="+e+" in method simpleExperiment"+e);
                e.printStackTrace();
                System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
                System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes"+test.numInstances()+" instances");

                System.exit(0);
        }
         return acc;
    }    


    
    
    public static void collateResults(){
//3. Merge classifier files into a single file with average accuracies
        //NEED TO REWRITE FOR TRAIN TEST DIFF
        String basePath="C:\\Users\\ajb\\Dropbox\\Results\\Forest\\";
        OutFile acc=new OutFile(basePath+"CombinedAcc.csv");
        for(String cls:classifiers){
            acc.writeString(","+cls);
        }
        acc.writeString("\n");
        InFile[] allTest=new InFile[classifiers.length];
        for(int i=0;i<allTest.length;i++){
            String p=basePath+classifiers[i]+"Test.csv";
            if(new File(p).exists()){
                allTest[i]=new InFile(p);
//                System.out.println("File "+p+" opened ok");
            }
            else
                allTest[i]=null;//superfluous
//             p=basePath+classifiers[i]+"//"+classifiers[i]+"Train.csv";
        }
        for(int i=0;i<files.length;i++){
            acc.writeString(files[i]+",");
            String prev="First";
            for(int j=0;j<allTest.length;j++){
                if(allTest[j]==null){
                    acc.writeString(",");
                }
                else{//Find mean
                    try{
                        String r=allTest[j].readLine();
                        String[] res=r.split(",");
                        double mean=0;
                        for(int k=1;k<res.length;k++){
                            mean+=Double.parseDouble(res[k]);
                        }
                        if(res.length>1){
                            acc.writeString((mean/(res.length-1))+",");
                        }
                        else{
                            acc.writeString(",");
                        }
                        prev=r;
                    }catch(Exception ex){
                        System.out.println("failed to read line: "+ex+" previous line = "+prev);
                        System.exit(0);
                    }
                }
            }
            acc.writeString("\n");
        }
        for(InFile  inf:allTest)
            if(inf!=null)
                inf.closeFile();
        
    }
/**
 * nos cases, nos features, nos classes, nos cases/**
 * nos cases, nos features, nos classes, nos cases
 */   

    public static void summariseData(){
        
        OutFile out=new OutFile(DataSets.problemPath+"SummaryInfo.csv");
        out.writeLine("problem,numCases,numAtts,numClasses");
        for(String str:files){
            File f=new File(DataSets.problemPath+str+"/"+str+".arff");
            if(f.exists()){
                Instances ins=ClassifierTools.loadData(DataSets.problemPath+str+"/"+str);
                out.writeLine(str+","+ins.numInstances()+","+(ins.numAttributes()-1)+","+ins.numClasses());
            }
            else
                out.writeLine(str+",,");
        }
    }
    public static void collateSVMParameters(){
        InFile c=new InFile("C:\\Research\\Papers\\2017\\ECML Standard Parameters\\Section 6 choosing parameters\\TunedSVMParameterC.csv");
        InFile g=new InFile("C:\\Research\\Papers\\2017\\ECML Standard Parameters\\Section 6 choosing parameters\\TunedSVMParameterGamma.csv");
        int[][] counts=new int[25][25];
        double[] vals={0.000015,0.000031,0.000061,0.000122,0.000244,0.000488,0.000977,0.001953,0.003906,0.007813,0.015625,0.031250,0.062500,0.125000,0.250000,0.500000,1.000000,2.000000,4.000000,8.000000,16.000000,32.000000,64.000000,128.000000,256.000000};
        for(int i=0;i<files.length;i++){
            String line=c.readLine();
            String gLine=g.readLine();
            String[] splitC=line.split(",");
            String[] splitG=gLine.split(",");
            System.out.print("\n Problem="+splitC[0]);
            int cPos=0,gPos;
            for(int j=1;j<splitC.length;j++){
                if(!splitC[j].equals("")){
                    //Look up
                    int k=0;
                    double v=Double.parseDouble(splitC[j]);
                    try{
                    while(vals[k]!=v)
                        k++;
                    cPos=k;
                    }catch(Exception e){
                        System.out.println(" EXCEPTION : ="+e+" v = "+v+" k="+k);
                    }
                    k=0;
                    v=Double.parseDouble(splitG[j]);
                    while(vals[k]!=v)
                        k++;
                    gPos=k;
                    counts[cPos][gPos]++;
                    
//                    System.out.print("c Pos="+cPos+" G pos ="+gPos);
                }
            }
        }
        OutFile svm=new OutFile("C:\\Research\\Papers\\2017\\ECML Standard Parameters\\Section 6 choosing parameters\\svmParaCounts.csv");
        for(int i=0;i<counts.length;i++){
            for(int j=0;j<counts[i].length;j++)
                svm.writeString(counts[i][j]+",");
            svm.writeString("\n");
        }
        
    }
    public static void baseTimingOperation(){
//Benchmark operation is building full rotation forest on 
        
        
    }
    
    public static void generateFileGroups(){
        OutFile allUCI=new OutFile("C://Data/allUCI.txt");
        OutFile allUEA=new OutFile("C://Data/allUEA.txt");
        OutFile allUCIR=new OutFile("C://Data/allUCIReversed.txt");
        OutFile allUEAR=new OutFile("C://Data/allUEAReversed.txt");
        
        OutFile smallUCI=new OutFile("C://Data/smallUCI.txt");
        OutFile smallUEA=new OutFile("C://Data/smallUEA.txt");
        OutFile mediumUCI=new OutFile("C://Data/mediumUCI.txt");
        OutFile mediumUEA=new OutFile("C://Data/mediumUEA.txt");
        OutFile largeUCI=new OutFile("C://Data/largeUCI.txt");
        OutFile largeUEA=new OutFile("C://Data/largeUEA.txt");
        for(String str: DataSets.tscProblems85)
            allUEA.writeLine(str);
        for(String str: DataSets.UCIContinuousFileNames)
            allUCI.writeLine(str);
        ArrayList<String> uea=new ArrayList<>();
        for(String str: DataSets.tscProblems85)
            uea.add(str);
        Collections.reverse(uea);
        for(String str: uea)
            allUEAR.writeLine(str);
        ArrayList<String> uci=new ArrayList<>();
        for(String str: DataSets.UCIContinuousFileNames)
            uci.add(str);
        Collections.reverse(uci);
        for(String str: uci)
            allUCIR.writeLine(str);
        
}
    public static void main(String[] args) throws Exception{
        
//Generate all file names
        generateFileGroups();
        
        System.exit(0);
        String source="C://Data/UCIContinuous/";
        String dest="//cmptscsvr.cmp.uea.ac.uk/ueatsc/UCITrainTestSplit/";
        File f=new File(dest);
        if(!f.isDirectory())
            f.mkdirs();
        generateTestTRainUCIFolds(source,dest);
//        generateAllUCIFolds(source,dest);
//        generateAllRepoFolds(source,dest);
        System.exit(0);
        
      boolean ucrData=true;
       files=DataSets.tscProblems85;
 //      collateResults(30,true,args);
//UCIRotFTimingExperiment();
  //             System.exit(0);
 //       collateTrain();
 //          DataSets.problemPath="C:/Data/TSC Problems/";
 //      timingNormalisation("c:/temp/benchmark.csv");
      

         classifiers=new String[]{"SVM"};
        String dir="RepoScripts";
        String jarFile="ClassifierExperiment";
     generateScripts(true,4000,jarFile,DataSets.tscProblems85,dir);
    generateScripts(false,4000,jarFile,DataSets.tscProblems85,dir);
 /*        dir="UCIScripts";
     generateScripts(true,4000,jarFile,UCIContinuousFileNames,dir);
    generateScripts(false,4000,jarFile,UCIContinuousFileNames,dir);
 */   System.exit(0);

//        collateTrainTestResults(30);

        if(ucrData)
            runTSCDataSet(args);
        else
            runUCIDataSet(args);
    }

    
    public static void runTSCDataSet(String[] args) {
        if(args.length>0){//Cluster run
            DataSets.problemPath=DataSets.clusterPath+"TSCProblems/";
            DataSets.resultsPath=DataSets.clusterPath+"Results/RepoResults/";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
            }
            April2017Experiments.singleClassifierAndFoldTrainTestSplit(args);
        }
        else{
            DataSets.problemPath=DataSets.dropboxPath+"TSC Problems/";
            DataSets.resultsPath=DataSets.dropboxPath+"Results/RepoResults/";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
            }

            String[] paras={"RandFCV","ItalyPowerDemand","1"};
//            paras[0]="RotFCV";
//            paras[2]="1";
            April2017Experiments.singleClassifierAndFoldTrainTestSplit(paras);            
            long t1=System.currentTimeMillis();
            for(int i=2;i<=11;i++){
                paras[2]=i+"";
                April2017Experiments.singleClassifierAndFoldSingleDataSet(paras);            
            }
            long t2=System.currentTimeMillis();
            paras[0]="RandFOOB";
            April2017Experiments.singleClassifierAndFoldSingleDataSet(paras);            
            long t3=System.currentTimeMillis();
            for(int i=2;i<=11;i++){
                paras[2]=i+"";
                April2017Experiments.singleClassifierAndFoldSingleDataSet(paras);            
            }
            long t4=System.currentTimeMillis();
            System.out.println("Standard = "+(t2-t1)+", Enhanced = "+(t4-t3));
            
       }        
    }
    
    
    
    public static void runUCIDataSet(String[] args) {
        if(args.length>0){//Cluster run
            DataSets.problemPath=DataSets.clusterPath+"UCIContinuous/";
            DataSets.resultsPath=DataSets.clusterPath+"Results/UCIResults/";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
            }
            April2017Experiments.singleClassifierAndFoldSingleDataSet(args);
        }
        else{
            DataSets.problemPath=DataSets.dropboxPath+"UCI Problems/";
            DataSets.resultsPath=DataSets.dropboxPath+"Results/UCIResults/";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
            }

            String[] paras={"","semeion","1"};
            DataSets.problemPath="C:/Data/UCI Problems/";
            DataSets.resultsPath=DataSets.dropboxPath+"Results/UCIResults/";
            File file =new File("C:\\Users\\ajb\\Dropbox\\Results\\UCIResults");
            paras[0]="RotFCV";
            paras[2]="1";
            April2017Experiments.singleClassifierAndFoldSingleDataSet(paras);            
            long t1=System.currentTimeMillis();
            for(int i=2;i<=11;i++){
                paras[2]=i+"";
                April2017Experiments.singleClassifierAndFoldSingleDataSet(paras);            
            }
            long t2=System.currentTimeMillis();
            paras[0]="EnhancedRotF";
            April2017Experiments.singleClassifierAndFoldSingleDataSet(paras);            
            long t3=System.currentTimeMillis();
            for(int i=2;i<=11;i++){
                paras[2]=i+"";
                April2017Experiments.singleClassifierAndFoldSingleDataSet(paras);            
            }
            long t4=System.currentTimeMillis();
            System.out.println("Standard = "+(t2-t1)+", Enhanced = "+(t4-t3));
            
       }        
    }


    public static void UCIRotFTimingExperiment() throws Exception{
//Restrict to those with over 40 attributes
        OutFile times=new OutFile("c:/temp/RotFUCITimes.csv");
        for(String problem:UCIContinuousFileNames){
//See whether we want to do this one
            if(problem.equals("miniboone")||problem.equals("connect-4"))
                continue;
            Instances inst=ClassifierTools.loadData("C:/Data/UCIContinuous/"+problem+"/"+problem);
            
            if(inst.numAttributes()-1>40){

                System.out.println(" Problem "+problem+" has "+(inst.numAttributes()-1)+" number of attributes");
                times.writeString(problem+","+(inst.numAttributes()-1)+","+(inst.numInstances())+",");
                RotationForest rot1=new RotationForest();
                rot1.setNumIterations(200);
                RotationForestLimitedAttributes rot2=new RotationForestLimitedAttributes();
                rot2.setNumIterations(200);
                rot2.tuneParameters(false);
                rot2.estimateAccFromTrain(false);
//Identical apart from this            
                rot2.setMaxNumAttributes(40);
                long t1=System.currentTimeMillis();
                rot1.buildClassifier(inst);
                long t2=System.currentTimeMillis();
                System.out.println(" Full RotF time = "+((t2-t1)/1000));
                times.writeString((t2-t1)+",");
                t1=System.currentTimeMillis();
                rot2.buildClassifier(inst);
                t2=System.currentTimeMillis();
                System.out.println(" truncated RotF time = "+((t2-t1)/1000));
                times.writeLine((t2-t1)+",");
                
                
                
            }            
        }
        
    }


    public static void UCRRotFTimingExperiment() throws Exception{
//Restrict to those with over 40 attributes
        OutFile times=new OutFile("c:/temp/RotFUCITimes.csv");
        for(String problem:DataSets.tscProblems85){
//See whether we want to do this one
            Instances inst=ClassifierTools.loadData("C:/Data/TSC Problems/"+problem+"/"+problem+"_TRAIN");
            if(problem.equals("HandOutlines"))
                continue;
            
            if(inst.numAttributes()-1>100){

                System.out.println(" Problem "+problem+" has "+(inst.numAttributes()-1)+" number of attributes");
                times.writeString(problem+","+(inst.numAttributes()-1)+","+(inst.numInstances())+",");
                RotationForest rot1=new RotationForest();
                rot1.setNumIterations(200);
                RotationForestLimitedAttributes rot2=new RotationForestLimitedAttributes();
                rot2.setNumIterations(200);
                rot2.tuneParameters(false);
                rot2.estimateAccFromTrain(false);
//Identical apart from this            
                rot2.setMaxNumAttributes(100);
                long t1=System.currentTimeMillis();
                rot1.buildClassifier(inst);
                long t2=System.currentTimeMillis();
                System.out.println(" Full RotF time = "+((t2-t1)/1000));
                times.writeString((t2-t1)+",");
                t1=System.currentTimeMillis();
                rot2.buildClassifier(inst);
                t2=System.currentTimeMillis();
                System.out.println(" truncated RotF time = "+((t2-t1)/1000));
                times.writeLine((t2-t1)+",");
                
                
                
            }            
        }
        
    }



}

