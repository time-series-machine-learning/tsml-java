/**
 *
 * @author ajb
 *local class to run experiments with the UCI data


*/
package papers.bagnall17defaultparas;

import development.DataSets;
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
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import vector_classifiers.TunedSVM;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.RotationForest;
import vector_classifiers.TunedRotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import vector_classifiers.TunedRandomForest;
import weka.core.Instances;


public class ClassifierComparison{
    public static String[] classifiers={"TunedSVM","TunedRandF","TunedRotF","Logistic","NB","IB1","C4.5","IBk","RandF","RotF","SVML","SVMQ","SVMRBF"};
    public static double propInTrain=0.5;
    public static int folds=30; 
    static String[] fileNames={//"abalone",
        "acute-inflammation","acute-nephritis","adult","annealing","arrhythmia","audiology-std","balance-scale","balloons","bank","blood","breast-cancer","breast-cancer-wisc","breast-cancer-wisc-diag","breast-cancer-wisc-prog","breast-tissue","car","cardiotocography-10clases","cardiotocography-3clases",
        //"chess-krvk","chess-krvkp",
        "congressional-voting","conn-bench-sonar-mines-rocks","conn-bench-vowel-deterding",
        //"connect-4",
        "contrac","credit-approval","cylinder-bands","dermatology","echocardiogram","ecoli","energy-y1","energy-y2","fertility","flags","glass","haberman-survival","hayes-roth","heart-cleveland","heart-hungarian","heart-switzerland","heart-va","hepatitis","hill-valley","horse-colic","ilpd-indian-liver","image-segmentation","ionosphere","iris","led-display","lenses","letter","libras","low-res-spect","lung-cancer","lymphography","magic","mammographic",
        //"miniboone",
        "molec-biol-promoter","molec-biol-splice","monks-1","monks-2","monks-3","mushroom","musk-1","musk-2","nursery","oocytes_merluccius_nucleus_4d","oocytes_merluccius_states_2f","oocytes_trisopterus_nucleus_2f","oocytes_trisopterus_states_5b","optical","ozone","page-blocks","parkinsons","pendigits","pima","pittsburg-bridges-MATERIAL","pittsburg-bridges-REL-L","pittsburg-bridges-SPAN","pittsburg-bridges-T-OR-D","pittsburg-bridges-TYPE","planning","plant-margin","plant-shape","plant-texture","post-operative","primary-tumor","ringnorm","seeds","semeion","soybean","spambase","spect","spectf","statlog-australian-credit","statlog-german-credit","statlog-heart","statlog-image","statlog-landsat","statlog-shuttle","statlog-vehicle","steel-plates","synthetic-control","teaching","thyroid","tic-tac-toe","titanic","trains","twonorm","vertebral-column-2clases","vertebral-column-3clases","wall-following","waveform","waveform-noise","wine","wine-quality-red","wine-quality-white","yeast","zoo"};
    static boolean debug=false;
//Parameter ranges for search, use same for C and gamma   
    static double[] svmParas={0.00390625, 0.015625, 0.0625, 0.25, 0.5, 1, 2, 4, 16, 256};
//Parameter ranges for trees for randF and rotF
    static int[] numTrees={10,50,100,200,300,400,500,600,700,800,900,1000,1250,1500,1750,2000};

    public static void generateScripts(boolean grace,int mem){
//Generates cluster scripts for allTest combos of classifier and data set
//Generates txt files to run jobs for a single classifier        
        String path=DataSets.dropboxPath+"Code\\Cluster Scripts\\UCIScripts\\";
        File f=new File(path);
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
                of.writeLine("java -jar CompareClassifiers.jar "+s+" "+a+" $LSB_JOBINDEX");                
                if(grace)
                    of2.writeLine("bsub < Scripts/UCIScripts/"+s+a+"Grace.bsub");
                else
                    list.add("bsub < Scripts/UCIScripts/"+s+a+".bsub");
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
        for(String str:fileNames){
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


    public static void collateResults(int folds, boolean onCluster){
        if(onCluster)
           DataSets.resultsPath=DataSets.clusterPath+"Results/UCIResults/";

        String basePath=DataSets.resultsPath;
//1. Collate single folds into single classifier_problem files        
        for(String cls:classifiers){
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
                for(int i=0;i<fileNames.length;i++){
                    String name=fileNames[i];
                    clsResults.writeString(fileNames[i]+",");
                    trainResults.writeString(fileNames[i]+",");
                    cPara.writeString(fileNames[i]+",");
                    gammaPara.writeString(fileNames[i]+",");
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
        for(String cls:classifiers){
            acc.writeString(","+cls);
            count.writeString(","+cls);
        }
        acc.writeString("\n");
        count.writeString("\n");
        InFile[] allTest=new InFile[classifiers.length];
        for(int i=0;i<allTest.length;i++){
            String p=basePath+classifiers[i]+"/"+classifiers[i]+"Test.csv";
            if(new File(p).exists())
                allTest[i]=new InFile(p);
            else
                allTest[i]=null;//superfluous
//             p=basePath+classifiers[i]+"//"+classifiers[i]+"Train.csv";
        }
        for(int i=0;i<fileNames.length;i++){
            acc.writeString(fileNames[i]+",");
            count.writeString(fileNames[i]+",");
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


    public static Classifier setClassifier(String classifier, int fold){
//RandF or RotF
        Classifier basic;
        switch(classifier){
            case "TunedRandF":
            TunedRandomForest randF = new TunedRandomForest();
            randF.setNumTreesRange(numTrees);
            randF.tuneParameters(false);  // just testing whether tuning the number of trees helps
            randF.debug(debug);
            randF.setSeed(fold);
            return randF;
            case "TunedRotF":
            TunedRotationForest rotF = new TunedRotationForest();
            rotF.tuneParameters(true);
            rotF.setNumTreesRange(numTrees);
            rotF.debug(debug);
            rotF.setSeed(fold);
           return rotF; 
            case "TunedSVM":
            TunedSVM svm = new TunedSVM();
            svm.optimiseKernel(false);
            svm.optimiseParas(true);
//            svm.setParaSpace(svmParas);
            svm.debug(debug);
            svm.setSeed(fold);
            return svm;
            case "Logistic":
                basic=new Logistic();
                break;
            case "IB1":
                basic=new IB1();
                break;
            case "NB":
                basic=new NaiveBayes();
                break;
            case "C4.5":
                basic = new J48();
                break;
            case "IBk":
                basic=new IBk();
                ((IBk)basic).setCrossValidate(true);
                break;
            case "RotF":
                basic=new RotationForest();
                break;
            case "RandF":
                basic=new RandomForest();
                break;
            case "SVML":
                basic=new SMO();
                PolyKernel poly1=new PolyKernel();
                poly1.setExponent(1);
                ((SMO)basic).setKernel(poly1);
                break;
            case "SVMQ":
                basic=new SMO();
                PolyKernel poly2=new PolyKernel();
                poly2.setExponent(2);
                ((SMO)basic).setKernel(poly2);
                break;
            case "SVMRBF":
                basic=new SMO();
                RBFKernel kernel = new RBFKernel();
                ((SMO)basic).setKernel(kernel);
                break;
            default:
            throw new RuntimeException("Unknown classifier");
        }
        return basic;
    }
        
    
    public static void singleClassifierAndFold(String[] args){
//first gives the problem file      
        String classifier=args[0];
        String problem=args[1];
        int fold=Integer.parseInt(args[2])-1;
   
        Classifier c=setClassifier(classifier,fold);
        Instances all=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem);
        all.randomize(new Random());
        
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
                System.out.println(" Error ="+e+" in method singleClassifierAndFold in class ClassifierComparison");
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
        for(int i=0;i<fileNames.length;i++){
            acc.writeString(fileNames[i]+",");
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
 * nos cases, nos features, nos classes, nos cases
 */   

    public static void summariseData(){
        
        OutFile out=new OutFile(DataSets.problemPath+"SummaryInfo.csv");
        out.writeLine("problem,numCases,numAtts,numClasses");
        for(String str:fileNames){
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
        for(int i=0;i<fileNames.length;i++){
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
    public static void main(String[] args) throws IOException{
        
       try
       {
           System.out.println((new File("C:\\Program Files\\Java\\jdk1.8.0_40\\bin")).toURI());
       }
       catch (Exception e)
       {
           e.printStackTrace();
       }
        
//       collateSVMParameters();
System.exit(0);
 //       collateTrain();
    generateScripts(true,8000);
    generateScripts(false,8000);
//    collateResults(30,true);
System.exit(0);
 
//        collateTrainTestResults(30);
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
            String[] paras={"TunedRandF","balloons","1"};
            
            singleClassifierAndFold(paras);            
        }
    }
}

