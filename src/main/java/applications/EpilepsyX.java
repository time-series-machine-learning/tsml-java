/*
Code to run EpilepsyX data using a leave one person out sampling
 */
package applications;

import development.DataSets;
import development.MatrixProfileExperiments;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import static papers.Bagnall16bakeoff.setClassifier;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import weka.classifiers.Classifier;
import timeseriesweka.classifiers.ensembles.SaveableEnsemble;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class EpilepsyX {
    public static String[] personID={"ID001","ID002","ID003","ID004","ID005","ID006"};
    static String[] classifiers={ //Benchmarks
        "ED", "RotF","DTW","EE","HESCA","TSF","ST",
        "BOSS",
        //Spectral
        "RISE",
        //Combos
        "FLATCOTE","HIVECOTE"};    
    
    static Instances[] sample(int person){//Returns the leave one person out train/test split
Instances all=ClassifierTools.loadData(DataSets.problemPath+"/EpilepsyX/EpilepsyX"); 
        Instances[] split=new Instances[2];
        split[0]=new Instances(all,0);
        split[1]=new Instances(all,0);
        for(Instance ins:all){
            if(ins.stringValue(0).equals(personID[person]))    //Stringvalues? Debug
                split[1].add(ins);
            else
                split[0].add(ins);
        }
//Remove the person ID        
        split[0].deleteAttributeAt(0);
        split[1].deleteAttributeAt(0);
        return split;
    }
    
    public static void collateResults(){
        String resultsPath="C:/Research/Results/EpilepsyXResults/";
        OutFile out=new OutFile(resultsPath+"collatedResults.csv");
        for(String c:classifiers)
            out.writeString(","+c);
        out.writeString("\n");
        for(int i=0;i<personID.length;i++){
            out.writeString(personID[i]);
            for(String c:classifiers){
                String p=resultsPath+c+"/Predictions/EpilepsyX/"+"testFold"+i+".csv";
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
    
    public static void singleClassifierAndPerson(String[] args){
//first gives the problem file      
        String classifier=args[0];
        int fold=Integer.parseInt(args[1])-1;
        Classifier c=MatrixProfileExperiments.setClassifier(classifier);
        Instances[] split=sample(fold); 
        File f=new File(DataSets.resultsPath+classifier);
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+classifier+"/Predictions";
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
        predictions=predictions+"/EpilepsyX";
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
//Check whether fold already exists, if so, dont do it, just quit
        f=new File(predictions+"/testFold"+fold+".csv");
        if(!f.exists() || f.length()==0){
      //      of.writeString(problem+","); );
//            if(c instanceof SaveParameterInfo)
//                ((SaveParameterInfo)c).setCVPath(predictions+"/trainFold"+fold+".csv");
            double acc =singleClassifierAndPerson(split[0],split[1],c,fold,predictions);
            System.out.println(classifier+","+"EpilepsyX"+","+fold+","+acc);
            
 //       of.writeString("\n");
        }
    }
    public static double singleClassifierAndPerson(Instances train, Instances test, Classifier c, int fold,String resultsPath){
        double acc=0;
        int act;
        int pred;
// Save internal info for ensembles
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
    public static void createEpilepsyScripts(boolean grace){
//Generates cluster scripts for all combos of classifier and data set
//Generates txt files to run jobs for a single classifier        
        String path="C:\\Users\\ajb\\Dropbox\\Code\\Cluster Scripts\\EpilepsyXScripts\\";
        File f=new File(path);
        if(!f.isDirectory())
            f.mkdir();
        int mem=4000;
            OutFile of2;
            if(grace)
                of2=new OutFile(path+"EpilepsyXGrace.txt");
            else
                of2=new OutFile(path+"EpilepsyX.txt");
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
            of.writeLine("#BSUB -J "+s+"[1-6]");
            of.writeLine("#BSUB -oo output/"+"EpilepsyX"+s+".out");
            of.writeLine("#BSUB -eo error/"+"EpilepsyX"+s+".err");
            if(grace){
                of.writeLine("#BSUB -R \"rusage[mem="+mem+"]\"");
                of.writeLine("#BSUB -M "+mem);
                of.writeLine(" module add java/jdk/1.8.0_31");
            }
            else{
                of.writeLine("#BSUB -R \"rusage[mem="+(4000+mem)+"]\"");
                of.writeLine("#BSUB -M "+(4000+mem));
                of.writeLine("module add java/jdk1.8.0_51");
            }
            of.writeLine("java -jar EpilepsyX.jar "+s+"  $LSB_JOBINDEX");                
            if(grace)
                of2.writeLine("bsub < Scripts/EpilepsyXScripts/"+s+"Grace.bsub");
            else
                of2.writeLine("bsub < Scripts/EpilepsyXScripts/"+s+".bsub");
        }   
    } 
  
    
    public static void main(String[] args){
        
//       collateResults();
//        createEpilepsyScripts(true);
//       createEpilepsyScripts(false);
//        System.exit(0);
        if(args.length>0){//Cluster run
            DataSets.problemPath=DataSets.clusterPath+"TSC Problems/";
            DataSets.resultsPath=DataSets.clusterPath+"Results/EpilepsyXResults/";
            singleClassifierAndPerson(args);
        }
        else{
            DataSets.problemPath=DataSets.dropboxPath+"TSC Problems/";
            DataSets.resultsPath=DataSets.dropboxPath+"Results/EpilepsyXResults/";
            String[] paras={"RotF","6"};
            singleClassifierAndPerson(paras);            
        }
    }    
    
    
}
