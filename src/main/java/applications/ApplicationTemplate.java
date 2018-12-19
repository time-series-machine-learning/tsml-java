/*
Template code for a single problem. Steps:

SET UP
1. Copy and rename this class <problemName>.java using Refactor -> Copy
2. In main, call setProblemName(<problemName>) 
3. In main, change these variables
       DataSets.dropboxPath="C:/Users/ajb/Dropbox/"; //Somewhere to put files locally. Doesnt have to be dropbox
        DataSets.clusterPath="/gpfs/home/ajb/";   //The cluster path, based on your username

SCRIPTS
3. Run this class to run the method createScripts. Verify the scripts are ok by reading them. 
4. On the cluster, create a Directory Scripts/. Inside this, create a Directory <problemName>Scripts/
5. Copy all over to this directory on the cluster
6. Copy the file <problemName>.txt into your cluster root directory
RUNNING A PROBLEM LOCALLY
DOES YOUR PROBLEM REQUIRE BESPOKE TRAIN/TEST SPLITS? If unclear, ask Tony 
No: Go to 9
Yes: Go to 7
7. Define the array sampleID with the splitting criteria. Look at EpilepsyX or talk to Tony if you are unsure
8. In main, uncomment the line setSampleByAttribute(true); 
9. In main, comment out the method createScripts and the System.exit(0). REMEMBER TO DO THIS!
10. Run this class. It should create a file system with a single fold of RotF. Debug to your satisfaction
RUNNING A PROBLEM ON THE CLUSTER
11. Clean and build your project. Remember to set the project executable class through project properties -> run to your class
12. Go to the dist folder in your project and copy the file TimeSeriesClassification.jar and rename <problemName>.jar
13. Copy problemName.jar into your root cluster directory
14. Open a Putty window, and try running a classifier 
bsub < Scripts/<problemName>Scripts/RotF.bsub
15. If you want to queue all classifiers, enter
sh <problemName>.txt
COLLATING RESULTS
16. Copy the whole results directory into 
DataSets.dropboxPath+"Results/<problemName>Results/
17. Uncomment out 
    collateResults();       
    System.exit(0);
18. This will create a file collatedResults.csv. Show this to Tony! 

repeat ad nauseam. 

*/
package applications;

import development.DataSets;
import papers.bagnall17defaultparas.RepoExperiments;
import development.MatrixProfileExperiments;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import static papers.Bagnall16bakeoff.setClassifier;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.Classifier;
import timeseriesweka.classifiers.ensembles.SaveableEnsemble;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class ApplicationTemplate {
/*This will dictate directory names and must match the arff file, which 
MUST be in the location    
DataSets.problemPath+"/"+problemName+"/"+problemName.arff    
    */
    public static String problemName="EthanolLevel";
    public static void setProblemName(String s){
        problemName=s;
    }
    
/**Set to true if the fold sampling method is set by a string attribute 
If it is true, the String attribute *must* be the first attribute, and 
the values it takes are dictated by the String array sampleID.
    
The number of folds equals sampleID.length    
    
**/
    static boolean sampleByAttribute=false;    
/** If you do not sample by a string attribute, you need to set  the proportion 
 * in the train fold
 */
   public static void setSampleByAttribute(boolean b){
       sampleByAttribute=b;
   }
    static double proportionInTrain=0.3;
/**
If you are performing a specific sampling, sampleID strings dictate the folds
Only used if sampleByAttribute set to true     
    **/
//<editor-fold defaultstate="collapsed" desc="sampleID string array: ">    
    public static final String[] sampleID = {        
        "aberfeldy",
        "aberlour", //difficult, seams/stickers in way
        "amrut",
        "ancnoc",
        "armorik",
        "arran10",
        "arran14",
        "asyla",
        "benromach",
        "bladnoch", //difficult, seams/stickers in way
        "blairathol",
        "exhibition",
        "glencadam",
        "glendeveron",
        "glenfarclas",
        "glengoyne",
        "glenlivet15",
        "glenmorangie",
        "glenmoray",
        "glenscotia",
        "oakcross",
        "organic",
        "peatmonster",
        "scapa", //difficult, seams/stickers in way
        "smokehead",
        "speyburn",
        "spicetree",
        "talisker"
    };
//</editor-fold>        
    
//<editor-fold defaultstate="collapsed" desc="Classifier string array:   ">     
    static String[] classifiers={ //Benchmarks
        "ED", "RotF","DTW","EE","HESCA","TSF","ST",
        "BOSS",
        //Spectral
        "RISE",
        //Combos
        "FLATCOTE","HIVECOTE"};    
//</editor-fold>      
    
    static Instances[] sample(int fold){//Returns the leave one bottle out train/test split
        Instances all=ClassifierTools.loadData(DataSets.problemPath+"/"+problemName+"/"+problemName); 
        Instances[] split=new Instances[2];
       
        
        if(sampleByAttribute){
            split[0]=new Instances(all,0);
            split[1]=new Instances(all,0);            
            for(Instance ins:all){
                if(ins.stringValue(0).equals(sampleID[fold]))    
                    split[1].add(ins);
                else
                    split[0].add(ins);
            }
    //Remove the bottle ID        
            split[0].deleteAttributeAt(0);
            split[1].deleteAttributeAt(0);
        }
        else{    //Just randomly stratify
            split=InstanceTools.resampleInstances(all, fold, proportionInTrain);
        }
        return split;
    }
    
    public static void collateResults(){
        String resultsPath=DataSets.resultsPath+problemName+"Results/";
        OutFile out=new OutFile(resultsPath+"collatedResults.csv");
        for(String c:classifiers)
            out.writeString(","+c);
        out.writeString("\n");
        for(int i=0;i<sampleID.length;i++){
            out.writeString(sampleID[i]);
            for(String c:classifiers){
                String p=resultsPath+c+"/Predictions/"+problemName+"/"+"testFold"+i+".csv";
                File f =new File(p);
                if(f.exists() && f.length()>0){//Could still fail
                    InFile inf=new InFile(p);
                    inf.readLine();
                    inf.readLine();
                    out.writeString(","+inf.readDouble());
                }
                else
                    out.writeString(",");
            }
            out.writeString("\n");
        }
        
    }
    
    public static void singleClassifierAndFold(String[] args){
//first gives the problem file      
        String classifier=args[0];
        int fold=Integer.parseInt(args[1])-1;
        Classifier c=MatrixProfileExperiments.setClassifier(classifier);
        Instances[] split=sample(fold); 
        String predictions=DataSets.resultsPath+classifier+"/Predictions";
        predictions=predictions+"/"+problemName;
        File f=new File(predictions);
        if(!f.exists())
            f.mkdirs();
//Check whether fold already exists, if so, dont do it, just quit
        f=new File(predictions+"/testFold"+fold+".csv");
        if(!f.exists() || f.length()==0){
      //      of.writeString(problem+","); );
            if(c instanceof TrainAccuracyEstimate)
                ((TrainAccuracyEstimate)c).writeCVTrainToFile(predictions+"/trainFold"+fold+".csv");
            double acc =ApplicationTemplate.singleClassifierAndFold(split[0],split[1],c,fold,predictions);
            System.out.println(classifier+","+problemName+","+fold+","+acc);
            
 //       of.writeString("\n");
        }
    }
    public static double singleClassifierAndFold(Instances train, Instances test, Classifier c, int fold,String resultsPath){
        double acc=0;
        int act;
        int pred;
// Save internal info for ensembles. 
        if(c instanceof SaveableEnsemble)
           ((SaveableEnsemble)c).saveResults(resultsPath+"/internalCV_"+fold+".csv",resultsPath+"/internalTestPreds_"+fold+".csv");
        try{              
            c.buildClassifier(train);
            StringBuilder str = new StringBuilder();
            DecimalFormat df=new DecimalFormat("##.######");
            
            for(int j=0;j<test.numInstances();j++)
            {
                act=(int)test.instance(j).classValue();
                test.instance(j).setClassMissing();
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
    public static void createScripts(boolean grace, int mem){
//Generates cluster scripts for all combos of classifier and data set
//Generates txt files to run jobs for a single classifier  
//Set up the dropboxPath where you want them        
        int folds=100;
        String path=DataSets.dropboxPath+"\\Cluster Scripts\\";
        path+=problemName+"Scripts\\";
        File f=new File(path);
        if(!f.isDirectory())
            f.mkdirs();
        OutFile of2;
        if(grace)
            of2=new OutFile(path+problemName+"Grace.txt");
        else
            of2=new OutFile(path+problemName+".txt");
        for(String s:classifiers){
            OutFile of;
            if(grace)
                of = new OutFile(path+s+"Grace.bsub");
            else
                of = new OutFile(path+s+".bsub");
            of.writeLine("#!/bin/csh");
            if(grace)
                of.writeLine("#BSUB -q short");
            else
                of.writeLine("#BSUB -q long-eth");
            if(sampleByAttribute)
                of.writeLine("#BSUB -J "+problemName+s+"[1-"+sampleID.length+"]");
            else
                of.writeLine("#BSUB -J "+problemName+s+"[1-"+folds+"]");
            of.writeLine("#BSUB -oo output/"+problemName+s+".out");
            of.writeLine("#BSUB -eo error/"+problemName+s+".err");
            if(grace){
                of.writeLine("#BSUB -R \"rusage[mem="+mem+"]\"");
                of.writeLine("#BSUB -M "+mem);
                of.writeLine(" module add java/jdk/1.8.0_31");
            }
            else{
                of.writeLine("#BSUB -R \"rusage[mem="+(mem)+"]\"");
                of.writeLine("#BSUB -M "+(mem));
                of.writeLine("module add java/jdk1.8.0_51");
            }
            of.writeLine("java -jar "+problemName+".jar "+s+"  $LSB_JOBINDEX");                
            if(grace)
                of2.writeLine("bsub < Scripts/"+problemName+"Scripts/"+s+"Grace.bsub");
            else
                of2.writeLine("bsub < Scripts/"+problemName+"Scripts/"+s+".bsub");
        }   
    } 
  
    
    public static void main(String[] args){
/**
 * Usage
 */        
//1. Must set this, and it must equal the arff name. ALL results will be 
//put in places based on this name        
        setProblemName("ItalyPowerDemand");
//If this is set to true, you must list the attribute names in the array
//sampleID. If you do not call this method, it will simply randomly sample train
//and test        
 //       setSampleByAttribute(true);  
//Set up file locations. 
        DataSets.dropboxPath="C:/Users/ajb/Dropbox/"; //Somewhere to put files locally. Doesnt have to be dropbox
        DataSets.clusterPath="/gpfs/home/ajb/";   //The cluster path, based on your username
// Create all the cluster scripts locally in a folder in DataSets.dropboxPath  
//It is up to you to then copy them over        
//True if using Grace, false if using HPC, second argument is amount of memory
//will default to 100 folds unless setSampleByAttribute set to true        
//        createScripts(true,4000);
       createScripts(false,6000);
//This creates a load of scripts you need to copy over. You can run individual classifiers with
// bsub < <ClusterLocation>/RotF.bsub
//If you copy the file  problemName.txt into your root, you can run all classifiers using
// sh < problemName.txt       
       
// Once you have results, copy them into DataSets.resultsPath+problemName+"Results/"
// then call this method
//        collateResults();
        
        System.exit(0);
        
        if(args.length>0){//Cluster run
            DataSets.problemPath=DataSets.clusterPath+"TSC Problems/";
            DataSets.resultsPath=DataSets.clusterPath+"Results/"+problemName+"Results/";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory())
                f.mkdir();
            singleClassifierAndFold(args);
        }
        else{ //Local run, do this first to debug
            DataSets.problemPath=DataSets.dropboxPath+"TSC Problems/";
            DataSets.resultsPath=DataSets.dropboxPath+"Results/"+problemName+"Results/";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory())
                f.mkdir();
            String[] paras={"RotF","6"};
            singleClassifierAndFold(paras);            
        }
    }    
    
    
}
