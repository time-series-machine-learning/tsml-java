/**
 *
 * @author ajb
 *local class to run experiments with the UCI data

* HESCA Improvements:
* Run, compare overall accuracy, compare difference in train/test accuracy.
* Run local timing experiment from HESCADevelopment
* 
* 1. Current HESCA: bug in that it does not rebuild after CV. Test one vs that.
* more or less done, can check further but think it makes no difference, except
* or to slow it down!
* version with rebuild now referred to as as simply HESCA
* 2. HESCAV1: Reduce the amount of CV with simple rule FOLDS1=10,FOLDS2=20
        * public static int findNumFolds(Instances train){
                int numFolds = train.numInstances();
                if(train.numInstances()>=300)
                    numFolds=FOLDS2;
                else if(train.numInstances()>=200 && train.numAttributes()>=200)
                    numFolds=FOLDS2;
                else if(train.numAttributes()>=600)
                    numFolds=FOLDS2;
                else if (train.numInstances()>=100) 
                    numFolds=FOLDS1;
                return numFolds;
            }
Results:
*   1. Is accuracy different?
* Simple test of difference of means, standard two classifier tests on means
*   2. Is the difference between train and test significantly different to zero for
* each, and if so, is the difference between differences significant?
* A bit more complex, or co
* 
* 2. HESCAV2: use RandomForest OOB  error instead of CV. Really just a prequil to doing the same with RotationForest
* although that

*/
package papers.bagnall17defaultparas;

import development.DataSets;
import development.HESCADevelopment.*;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.text.DecimalFormat;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import vector_classifiers.TunedSVM;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.RotationForest;
import vector_classifiers.CAWPE;
import timeseriesweka.classifiers.ensembles.SaveableEnsemble;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;


public class UCIExperiments{

    public static String[] classifiers={"1NN","SVML_0","SVMQ_0","SVMRBF_0","SVM_0ptKernel"};
    public static double propInTrain=0.5;
    static String[] fileNames={"abalone","acute-inflammation","acute-nephritis","adult","annealing","arrhythmia","audiology-std","balance-scale","balloons","bank","blood","breast-cancer","breast-cancer-wisc","breast-cancer-wisc-diag","breast-cancer-wisc-prog","breast-tissue","car","cardiotocography-10clases","cardiotocography-3clases","chess-krvk","chess-krvkp","congressional-voting","conn-bench-sonar-mines-rocks","conn-bench-vowel-deterding","connect-4","contrac","credit-approval","cylinder-bands","dermatology","echocardiogram","ecoli","energy-y1","energy-y2","fertility","flags","glass","haberman-survival","hayes-roth","heart-cleveland","heart-hungarian","heart-switzerland","heart-va","hepatitis","hill-valley","horse-colic","ilpd-indian-liver","image-segmentation","ionosphere","iris","led-display","lenses","letter","libras","low-res-spect","lung-cancer","lymphography","magic","mammographic","miniboone","molec-biol-promoter","molec-biol-splice","monks-1","monks-2","monks-3","mushroom","musk-1","musk-2","nursery","oocytes_merluccius_nucleus_4d","oocytes_merluccius_states_2f","oocytes_trisopterus_nucleus_2f","oocytes_trisopterus_states_5b","optical","ozone","page-blocks","parkinsons","pendigits","pima","pittsburg-bridges-MATERIAL","pittsburg-bridges-REL-L","pittsburg-bridges-SPAN","pittsburg-bridges-T-OR-D","pittsburg-bridges-TYPE","planning","plant-margin","plant-shape","plant-texture","post-operative","primary-tumor","ringnorm","seeds","semeion","soybean","spambase","spect","spectf","statlog-australian-credit","statlog-german-credit","statlog-heart","statlog-image","statlog-landsat","statlog-shuttle","statlog-vehicle","steel-plates","synthetic-control","teaching","thyroid","tic-tac-toe","titanic","trains","twonorm","vertebral-column-2clases","vertebral-column-3clases","wall-following","waveform","waveform-noise","wine","wine-quality-red","wine-quality-white","yeast","zoo"};

    
    public static void generateScripts(boolean grace,int mem){
//Generates cluster scripts for allTest combos of classifier and data set
//Generates txt files to run jobs for a single classifier        
        String path=DataSets.dropboxPath+"Code\\Cluster Scripts\\UCIScripts\\";
        
        File f=new File(path);
        int folds=30; 
        
        if(!f.isDirectory())
            f.mkdirs();
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
                of.writeLine("java -jar UCI.jar "+s+" "+a+" $LSB_JOBINDEX");                
                if(grace)
                    of2.writeLine("bsub < Scripts/UCIScripts/"+s+a+"Grace.bsub");
                else
                    of2.writeLine("bsub < Scripts/UCIScripts/"+s+a+".bsub");
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
 * @param cls
 */    
    
    public static void compareTrainToTestAcc(String cls){

        
        
    }
    
    
    public static void collateResults(int folds){
        String basePath="C:\\Research\\Results\\UCIResults\\";
//1. Collate single folds into single classifier_problem files        
        for(String cls:classifiers){
//Check classifier directory exists. 
            File f=new File(basePath+cls);
            if(f.isDirectory()){
//Write collated results for this classifier to a single file                
                OutFile clsResults=new OutFile(basePath+cls+"//"+cls+"Test.csv");
                OutFile missing=null;
                int missingCount=0;
                for(int i=0;i<fileNames.length;i++){
                    String name=fileNames[i];
                    clsResults.writeString(fileNames[i]+",");
                    String path=basePath+cls+"//Predictions//"+fileNames[i];
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
                            InFile inf=new InFile(path+"//testFold"+j+".csv");
                            inf.readLine();
                            inf.readLine();
                            clsResults.writeString(inf.readDouble()+",");    
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
        OutFile acc=new OutFile(basePath+"CombinedAcc.csv");
        OutFile count=new OutFile(basePath+"CombinedCount.csv");
        OutFile diff=new OutFile(basePath+"TrainTestDiff.csv");
        for(String cls:classifiers){
            acc.writeString(","+cls);
            count.writeString(","+cls);
            diff.writeString(","+cls);
        }
        acc.writeString("\n");
        count.writeString("\n");
        diff.writeString("\n");
        InFile[] allTest=new InFile[classifiers.length];
        for(int i=0;i<allTest.length;i++){
            String p=basePath+classifiers[i]+"//"+classifiers[i]+"Test.csv";
            if(new File(p).exists())
                allTest[i]=new InFile(p);
            else
                allTest[i]=null;//superfluous
             p=basePath+classifiers[i]+"//"+classifiers[i]+"Train.csv";
        }
        for(int i=0;i<fileNames.length;i++){
            acc.writeString(fileNames[i]+",");
            count.writeString(fileNames[i]+",");
            for(int j=0;j<allTest.length;j++){
                if(allTest[j]==null){
                    acc.writeString(",");
                    count.writeString("0,");
                }
                else{//Find mean
                    String[] res=allTest[j].readLine().split(",");
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
                }
            } 
            acc.writeString("\n");
            count.writeString("\n");
            
        }
        
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
                for(int i=0;i<fileNames.length;i++){
                    String name=fileNames[i];
                    clsResults.writeString(fileNames[i]+",");
                    String path=basePath+cls+"//Predictions//"+fileNames[i];
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
        for(int i=0;i<fileNames.length;i++){
            diff.writeString(fileNames[i]+",");
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


    public static Classifier setClassifier(String classifier){
        Classifier c=null;
        switch(classifier){
//TIME DOMAIN CLASSIFIERS            
            case "IBk":
                c=new IBk();
                ((IBk)c).setCrossValidate(true);
                break;

            case "SVMRBF_0":
                TunedSVM c2=new TunedSVM();
                c2.optimiseKernel(false);
                c2.optimiseParas(true);
                RBFKernel kernel2 = new RBFKernel();
                c2.setKernel(kernel2);
                c=c2;
               break;
            case "SVML_0":
                TunedSVM c3=new TunedSVM();
                c3.optimiseKernel(false);
                c3.optimiseParas(true);
                PolyKernel p=new PolyKernel();
                p.setExponent(1);
                c3.setKernel(p);
                c=c3;
               break;
            case "SVMQ_0":
                TunedSVM c4=new TunedSVM();
                c4.optimiseKernel(false);
                c4.optimiseParas(true);
                PolyKernel p2=new PolyKernel();
                p2.setExponent(2);
                c4.setKernel(p2);
                c=c4;
               break;
            case "SVM_0ptKernel":
                TunedSVM c5=new TunedSVM();
                c5.optimiseKernel(true);
                c=c5;
               break;
            case "HESCA":
                c=new CAWPE();
                break;
            case "C45":
                c=new J48();
                break;
            case "NB":
                c=new NaiveBayes();
                break;
            case "SVML":
                c=new SMO();
                PolyKernel p3=new PolyKernel();
                p3.setExponent(1);
                ((SMO)c).setKernel(p3);
                break;
            case "SVMQ":
                c=new SMO();
                PolyKernel p4=new PolyKernel();
                p4.setExponent(2);
                ((SMO)c).setKernel(p4);
                break;
            case "BN":
                c=new BayesNet();
                break;
            case "MLP":
                c=new MultilayerPerceptron();
                break;
            case "RandF":
                c= new RandomForest();
                ((RandomForest)c).setNumTrees(500);
                break;
            case "RotF10":
                c= new RotationForest();
                ((RotationForest)c).setNumIterations(10);
                break;
            case "RotF25":
                c= new RotationForest();
                ((RotationForest)c).setNumIterations(25);
                break;
            case "RotF50": case "RotF":
                c= new RotationForest();
                ((RotationForest)c).setNumIterations(50);
                break;
            case "Logistic":
                c= new Logistic();
                break;
           default:
                System.out.println("WTF? UNKNOWN CLASSIFIER "+classifier+" Not implemented ");
                System.exit(0);
//                throw new Exception("Unknown classifier "+classifier);
        }
        return c;
    }
        
    
    public static void singleClassifierAndFold(String[] args){
//first gives the problem file      
        String classifier=args[0];
        String problem=args[1];
        int fold=Integer.parseInt(args[2])-1;
   
        Classifier c=setClassifier(classifier);
        Instances all=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem);
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
            double acc =singleClassifierAndFold(split[0],split[1],c,fold,predictions);
            System.out.println(classifier+","+problem+","+fold+","+acc);
            
 //       of.writeString("\n");
        }
    }
    
    public static double singleClassifierAndFold(Instances train, Instances test, Classifier c, int fold,String resultsPath){
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
                str.append(",,");
                for(double d:probs){
                    str.append(df.format(d));
                    str.append(",");
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
                System.out.println(" Error ="+e+" in method singleClassifierAndFold in class UCIExperiments");
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

   
    public static void main(String[] args) throws IOException{
 //      generateScripts(true,4000);
//       generateScripts(false,4000);
//       System.exit(0);
//        collateResults(30);
//        collateTrainTestResults(30);
//      System.exit(0);
 /*       Runtime rt=Runtime.getRuntime();
        Process proc=rt.exec("echo $USER");
        OutputStream os=proc.getOutputStream();
        String str=os.toString();
        System.out.println(" OUTPUT STREAM = "+str+" IT SHOULD BE ajb");
 */       if(args.length>0){//Cluster run
            DataSets.problemPath=DataSets.clusterPath+"UCIContinuous/";
            DataSets.resultsPath=DataSets.clusterPath+"Results/UCIResults/";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
            }
            singleClassifierAndFold(args);
        }
        else{
            DataSets.problemPath=DataSets.dropboxPath+"UCI Problems/";
            DataSets.resultsPath=DataSets.dropboxPath+"Results/UCIResults/";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
            }
            String[] paras={"SVM_0ptKernel","monks-1","5"};
            
            singleClassifierAndFold(paras);            
        }
    }
}
