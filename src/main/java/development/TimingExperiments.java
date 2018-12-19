/**
* 
*/
package development;

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
import utilities.ClassifierResults;
import utilities.CrossValidator;
import weka.classifiers.trees.RandomForest;
import vector_classifiers.TunedRandomForest;
import weka.classifiers.functions.LinearRegression;
import static weka.classifiers.functions.LinearRegression.TAGS_SELECTION;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.NormalizeAttribute;


public class TimingExperiments{
    static boolean debug=true;
    public static String[] classifiers={"RotF","RandRotF"};
 
    public static String problemPath="C:/Data/";
    public static String resultsPath="C:/Temp/";
    public static int folds=1;
    String dataSet="UCI";
    public static void main(String[] args) throws Exception{
//        System.out.println("Rescale time experiments ="+rescaleTimeModel());
//        System.exit(0);
        randomDataTimingExperiment("RotF");
//        regressionModel();
//        timingExperiment("C45","UCI");
//        timingExperiment("C45","UEA");
//        timingExperiment("C45","ST");
//        timingExperiment("RotF","UCI");
//        timingExperiment("RotF","UEA");
//        timingExperiment("RotF","ST");

    }
    public static boolean ignoreThisFile(String str){
        File f=new File(str+".arff");
//        System.out.println(problemPath+str+"/"+str);
        if(!f.exists()||f.length()==0)
            return true;
/*UCI Ignore
        if(str.equals("miniboone")||str.equals("connect-4"))
            return true;        
        else if(str.equals("handOutlines"))
            return true;
 //UCR ignore pigs      
  */      
        return false;
        
    }
    public static void timingExperiment(String classifierName,String dataSet) throws Exception{
//Restrict to those with over 40 attributes
        String[] files; 
        String pPath;
        if(dataSet.equals("UCI")){
            pPath=problemPath+"UCIContinuous/";
            files=DataSets.UCIContinuousFileNames;
        }
        else if(dataSet.equals("UEA")){
            pPath=problemPath+"TSCProblems/";
            files=DataSets.tscProblems46;
            System.out.println("Doing UEA problems");
        }
        else if(dataSet.equals("ST")){
            pPath=problemPath+"BalancedClassShapeletTransform/";
            files=DataSets.tscProblems46;
        }
        else
            throw new Exception(dataSet+" not known");
        OutFile times=new OutFile(resultsPath+classifierName+dataSet+"Times.csv");
        for(String problem:files){
//See whether we want to do this one
            String fileName=pPath+problem+"/"+problem;
                if(dataSet.equals("ST"))
                    fileName+="0_TRAIN";
            if(ignoreThisFile(fileName)){
                System.out.println("Ignoring :"+problem);
                continue;
            }
            try{
                Instances inst;
                inst=ClassifierTools.loadData(fileName);

                if(inst.numAttributes()-1>40){
                    long[] t=new long[folds];
                    times.writeString(problem+","+(inst.numAttributes()-1)+","+(inst.numInstances())+",");
                    for(int i=0;i<folds;i++){

                        Classifier c=Experiments.setClassifier(classifierName, i);
                        System.out.println(" Problem "+problem+" has "+(inst.numAttributes()-1)+" number of attributes");
                        if(dataSet.equals("ST"))
                            inst=ClassifierTools.loadData(pPath+problem+"/"+problem+i+"_TRAIN");
                        
                        long t1=System.currentTimeMillis();
                        c.buildClassifier(inst);
                        long t2=System.currentTimeMillis();
                        t[i]=(t2-t1);
                        times.writeString(","+t[i]);                
                        System.out.println(problem+" "+classifierName+" time = "+((t2-t1)));                
                    }

                    times.writeString("\n");                

                }
                else
                    System.out.println("Ignoring "+problem+" num attributes ="+(inst.numAttributes()-1));
            }catch(Exception e){
                System.out.println("File "+problemPath+problem+"/"+problem+" not present");
            }
        }               
    }


    public static void randomDataTimingExperiment(String classifierName) throws Exception{

        OutFile times=new OutFile(resultsPath+classifierName+"RandomRotFDataTimes.csv");
        for(int n=100;n<=1000;n+=100){
            for(int m=100;m<=1000;m+=100){
    //See whether we want to do this one
                try{
                    Instances inst;                    
                    inst=ClassifierTools.generateRandomProblem(n,m,4);
                    long[] t=new long[folds];
                    times.writeString((inst.numAttributes()-1)+","+(inst.numInstances())+",");
                    for(int i=0;i<folds;i++){

                        Classifier c=Experiments.setClassifier(classifierName, i);
//                        System.out.println(" Problem "+problem+" has "+(inst.numAttributes()-1)+" number of attributes");

                        long t1=System.currentTimeMillis();
                        c.buildClassifier(inst);
                        long t2=System.currentTimeMillis();
                        t[i]=(t2-t1);
                        times.writeString(","+t[i]);                
                        System.out.println("m="+m+" n ="+n+" "+classifierName+" time = "+((t2-t1)));                
                    }

                    times.writeString("\n");                

                }catch(Exception e){
                    System.out.println("Exception ="+e+" m ="+m+" n ="+n);
                    e.printStackTrace();
                    System.exit(0);
                }
            }               
        }
    }
    public static long benchmarkTime17AJBPC=8876;
    public static long benchmarkTimeLaptop=16367;
    public static long benchmarkTime2209324=10000;
    public static long benchmarkTime2264419=10960;
    public static double rescaleTimeModel(){
        int size=1000;
        double sum=0;
        Random rng= new Random();
        ArrayList<Long> times=new ArrayList<>();
        for(int t=0;t<11;t++){
            long t1=System.currentTimeMillis();
            for(int i=0;i<size;i++)
                for (int j = 0; j < size; j++) {
                    for (int k = 0; k < size; k++) {
                        sum+=rng.nextInt();
                    }
                }
            long t2=System.currentTimeMillis();
            System.out.println("Time taken to add up lots of  ints ="+(t2-t1)+" sum is "+sum);
            times.add(t2-t1);
        }
        Collections.sort(times);
        double median=times.get(5);
        System.out.println("Median time ="+median);
        double scale=median/(double)benchmarkTime2264419;
        
        return scale;
    }
    
    
    public static void regressionModel() throws Exception{
        Instances data = ClassifierTools.loadData("C:\\Research\\Papers\\2017\\JMLR Rotation Forest\\C45LogModel");
        NormalizeAttribute na=new NormalizeAttribute(data);
 //       data=na.process(data);
        LinearRegression linReg=new LinearRegression();
 //       linReg.setRidge(1.0);
        Random rng = new Random();
        rng.setSeed(0);
        
//        data.randomize(rng);
        linReg.setAttributeSelectionMethod(new SelectedTag(1,TAGS_SELECTION));
        OutFile out=new OutFile("C:\\Research\\Papers\\2017\\JMLR Rotation Forest\\C45ResultsLog.csv");
        linReg.setDebug(true);
        for(int i=0;i<data.numInstances();i++){
//Build fold i
            Instances train = new Instances(data);
            Instance test;
            test=train.remove(i);
            linReg.buildClassifier(train);
            double pred=linReg.classifyInstance(test);
            out.writeLine(i+","+test.classValue()+","+pred);
            System.out.println(i+","+test.classValue()+","+pred);
            
        }
        
    
    }

 
    public static void UCIRotFTimingExperiment() throws Exception{
//Restrict to those with over 40 attributes
        OutFile times=new OutFile("c:/temp/RotFUCITimes.csv");
        for(String problem:DataSets.UCIContinuousFileNames){
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

