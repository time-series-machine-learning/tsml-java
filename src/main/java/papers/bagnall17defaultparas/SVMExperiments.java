/**
 *
 * @author ajb
 *local class to run experiments with the UCI data on SVM classifiers

* Most of this code is to help run things on the cluster. It generates a lot 
* of results which may not be necessary. 
* 
* For the simple usecase, read the method generateAll

*/
package papers.bagnall17defaultparas;

import development.DataSets;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.Classifier;
import vector_classifiers.TunedSVM;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.IB1;
import weka.core.Instances;


public class SVMExperiments{

    public static String[] classifiers={"1NN","SVML","SVMQ","SVMRBF","SVMLTuned","SVMQTuned","SVMRBFTuned"};
    public static double propInTrain=0.5;
    public static int folds=30; 
    static String[] fileNames={"abalone","acute-inflammation","acute-nephritis","adult","annealing","arrhythmia","audiology-std","balance-scale","balloons","bank","blood","breast-cancer","breast-cancer-wisc","breast-cancer-wisc-diag","breast-cancer-wisc-prog","breast-tissue","car","cardiotocography-10clases","cardiotocography-3clases","chess-krvk","chess-krvkp","congressional-voting","conn-bench-sonar-mines-rocks","conn-bench-vowel-deterding","connect-4","contrac","credit-approval","cylinder-bands","dermatology","echocardiogram","ecoli","energy-y1","energy-y2","fertility","flags","glass","haberman-survival","hayes-roth","heart-cleveland","heart-hungarian","heart-switzerland","heart-va","hepatitis","hill-valley","horse-colic","ilpd-indian-liver","image-segmentation","ionosphere","iris","led-display","lenses","letter","libras","low-res-spect","lung-cancer","lymphography","magic","mammographic","miniboone","molec-biol-promoter","molec-biol-splice","monks-1","monks-2","monks-3","mushroom","musk-1","musk-2","nursery","oocytes_merluccius_nucleus_4d","oocytes_merluccius_states_2f","oocytes_trisopterus_nucleus_2f","oocytes_trisopterus_states_5b","optical","ozone","page-blocks","parkinsons","pendigits","pima","pittsburg-bridges-MATERIAL","pittsburg-bridges-REL-L","pittsburg-bridges-SPAN","pittsburg-bridges-T-OR-D","pittsburg-bridges-TYPE","planning","plant-margin","plant-shape","plant-texture","post-operative","primary-tumor","ringnorm","seeds","semeion","soybean","spambase","spect","spectf","statlog-australian-credit","statlog-german-credit","statlog-heart","statlog-image","statlog-landsat","statlog-shuttle","statlog-vehicle","steel-plates","synthetic-control","teaching","thyroid","tic-tac-toe","titanic","trains","twonorm","vertebral-column-2clases","vertebral-column-3clases","wall-following","waveform","waveform-noise","wine","wine-quality-red","wine-quality-white","yeast","zoo"};

/**
 * This method is for guidance really, we do not recommend trying to run all 
 * experiments in a single thread unless you have a lot of time ... we distribute
 * it over 1000 jobs.
 */    
    public static void generateAll(){
//Set this to the location of the problem files. Each problem needs to be in 
//its own directory of the same name, e.g.C:\Users\ajb\Dropbox\UCI Problems\abalone\abalone.arff       
        DataSets.problemPath=DataSets.dropboxPath+"UCI Problems/";
//Where to write results. It will write a separate file for each fold.            
        DataSets.resultsPath=DataSets.dropboxPath+"Results/UCIResults/";
        File f=new File(DataSets.resultsPath);
        if(!f.isDirectory()){
            f.mkdirs();
        }
        folds=30;
        propInTrain=0.5;
        
        for(String cls:classifiers){
            for(String prob:fileNames){
                for(int i=0;i<folds;i++){
                    String[] paras={cls,prob,i+""};
                    singleClassifierAndFold(paras);            
                }
            }
        }
        
    }
    
   
/**
 * 
 * Generates cluster scripts for all combos of classifier and data set
**/    
    public static void generateScripts(boolean grace,int mem){
        String path=DataSets.dropboxPath+"Code\\Cluster Scripts\\UCIScripts\\";
        File f=new File(path);
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
                of.writeLine("java -jar SVM.jar "+s+" "+a+" $LSB_JOBINDEX");                
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
                                        gammaPara.writeString(Double.parseDouble(trainRes[3])+",");
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
            String p=basePath+classifiers[i]+"//"+classifiers[i]+"Test.csv";
            if(new File(p).exists())
                allTest[i]=new InFile(p);
            else
                allTest[i]=null;//superfluous
//             p=basePath+classifiers[i]+"//"+classifiers[i]+"Train.csv";
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
            for(InFile  inf:allTest)
                if(inf!=null)
                    inf.closeFile();
            acc.writeString("\n");
            count.writeString("\n");
            
            
        }
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


    public static Classifier setClassifier(String classifier){
        Classifier c=null;
        RBFKernel kernel1;
        TunedSVM c2;
        PolyKernel kernel2;
        switch(classifier){
            case "1NN":
                c=new IB1();
               break;
            case "SVMRBFTuned":
                c2=new TunedSVM();
                c2.optimiseKernel(false);
                c2.optimiseParas(true);
                kernel1= new RBFKernel();
                c2.setKernel(kernel1);
                c=c2;
               break;
            case "SVMLTuned":
                c2=new TunedSVM();
                c2.optimiseKernel(false);
                c2.optimiseParas(true);
                kernel2=new PolyKernel();
                kernel2.setExponent(1);
                c2.setKernel(kernel2);
                c=c2;
               break;
            case "SVMQTuned":
                c2=new TunedSVM();
                c2.optimiseKernel(false);
                c2.optimiseParas(true);
                kernel2=new PolyKernel();
                kernel2.setExponent(2);
                c2.setKernel(kernel2);
                c=c2;
               break;
            case "SVMTunedKernel":
                c2=new TunedSVM();
                c2.optimiseKernel(true);
                c=c2;
               break;
            case "SVML":
                c=new SMO();
                kernel2=new PolyKernel();
                kernel2.setExponent(1);
                ((SMO)c).setKernel(kernel2);
                break;
            case "SVMQ":
                c=new SMO();
                kernel2=new PolyKernel();
                kernel2.setExponent(2);
                ((SMO)c).setKernel(kernel2);
                break;
            case "SVMRBF":
                c=new SMO();
                kernel1= new RBFKernel();
                ((SMO)c).setKernel(kernel1);
                break;
           default:
                System.out.println("WTF? UNKNOWN SVM CLASSIFIER: "+classifier+"  not an option in SVMExperiments");
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
 //     generateScripts(true,2000);
       generateScripts(false,10000);
 //      System.exit(0);
//        collateResults(30,true);
//        collateTrainTestResults(30);
      System.exit(0);
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
