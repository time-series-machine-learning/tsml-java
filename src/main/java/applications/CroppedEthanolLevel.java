package applications;

import development.DataSets;
import development.MatrixProfileExperiments;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.Classifier;
import timeseriesweka.classifiers.ensembles.SaveableEnsemble;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Application code for the CroppedEthanol4Class problem
 * 
 * @author James Large
 */
public class CroppedEthanolLevel {
/*This will dictate directory names and must match the arff file, which 
MUST be in the location    
DataSets.problemPath+"/"+problemName+"/"+problemName.arff    
    */
    public static String problemName="CroppedEthanolLevel";
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
    
    static int[] classifierMems={ //Benchmarks
        2000, 2000,2000,3000,4000,4000,4000,
        4000,
        //Spectral
        4000,
        //Combos
        4000,4000};  
    
    static String[] classifierQueues={ //Benchmarks
        "short", "short","short","long","short","short","short",
        "short",
        //Spectral
        "short",
        //Combos
        "long","long"};  
    
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
        OutFile out=new OutFile(resultsPath+problemName + "Collated.csv");
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
            double acc =CroppedEthanolLevel.singleClassifierAndFold(split[0],split[1],c,fold,predictions);
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
    public static void createScripts(boolean grace){
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
        
        String outfileTag = "_%I";
                
        for(int i = 0; i < classifiers.length; ++i){
            String s = classifiers[i];
            int mem = classifierMems[i];
            OutFile of;
            if(grace)
                of = new OutFile(path+s+"Grace.bsub");
            else
                of = new OutFile(path+s+".bsub");
            of.writeLine("#!/bin/csh");
            
            if(grace)
                of.writeLine("#BSUB -q " + classifierQueues[i]);
            else
                of.writeLine("#BSUB -q "+classifierQueues[i]+"-eth");
            
            if(sampleByAttribute)
                of.writeLine("#BSUB -J "+problemName+s+"[1-"+sampleID.length+"]");
            else
                of.writeLine("#BSUB -J "+problemName+s+"[1-"+folds+"]");
            
            of.writeLine("#BSUB -oo "+problemName+"/output/"+problemName+s+outfileTag+".out");
            of.writeLine("#BSUB -eo "+problemName+"/error/"+problemName+s+outfileTag+".err");
            
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
            of.writeLine("java -Xmx"+ mem +"m -jar "+problemName+"/"+problemName+".jar "+s+"  $LSB_JOBINDEX " + problemName);                
            if(grace)
                of2.writeLine("bsub < "+problemName+"/Scripts/"+s+"Grace.bsub");
            else
                of2.writeLine("bsub < "+problemName+"/Scripts/"+s+".bsub");
        }   
    } 
  
    
    public static void main(String[] args){   
                System.out.println("CroppedEthanolLevel Run");
        
        setProblemName("CroppedEthanol4Class_AllBottles");    
        setSampleByAttribute(true);  
        
        DataSets.dropboxPath="C:/JamesLPHD/AlcoholExps/"; //Somewhere to put files locally. Doesnt have to be dropbox
        DataSets.clusterPath="/gpfs/home/xmw13bzu/";   //The cluster path, based on your username   
        DataSets.resultsPath=DataSets.dropboxPath+"Results/";   //The cluster path, based on your username   
                
//        createScripts(true);
//        collateResults();
        
//        System.exit(0);
        
        if(args.length>0){//Cluster run
            setProblemName(args[2]);    
            DataSets.problemPath=DataSets.clusterPath+"TSC Problems/";
            DataSets.resultsPath=DataSets.clusterPath+"Results/"+problemName+"Results/";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory())
                f.mkdir();
            singleClassifierAndFold(args);
        }
        else{ //Local run, do this first to debug
            DataSets.problemPath="C:/TSC Problems/";
            DataSets.resultsPath=DataSets.dropboxPath+"Results/"+problemName+"Results/";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory())
                f.mkdir();
            String[] paras={"ED","6"};
            singleClassifierAndFold(paras);            
        }
    }    
    
    
}
