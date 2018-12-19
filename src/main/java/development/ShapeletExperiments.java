/*
Experiment to test the effect of shapelet sampling 
Fix casesPerClass to {50,50}
Fix Model.error to 1
Shapelet length to 29
Proportion train/test to 50/50

Increase lengths for the data
for 100 to 1000 in 100's
for each length, do 100 resamples for ShapeletTransformClassifier with time limit set to
1 minute
1 hour 
1 day

Save and plot to find the point at which sampling makes it significantly worse 


*/
package development;

import fileIO.OutFile;
import statistics.simulators.Model;
import statistics.simulators.SimulateShapeletData;
import timeseriesweka.classifiers.ShapeletTransformClassifier;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class ShapeletExperiments {
    static int[] casesPerClass=new int[]{50,50};
    static double trainProp=0.5;
    
    public static void shapeletSimulatorWithLength(int seriesLength, int fold){
        Model.setDefaultSigma(1);
//            "EE","HESCA","TSF","TSBF","FastShapelets","ST","LearnShapelets","BOP","BOSS","RISE","COTE"};
        OutFile of=new OutFile(DataSets.resultsPath+"testAcc_"+seriesLength+"_"+fold+".csv");
//Generate data
        Model.setGlobalRandomSeed(fold);
        Instances data=SimulateShapeletData.generateShapeletData(seriesLength,casesPerClass);
//Split data
        Instances[] split=InstanceTools.resampleInstances(data, fold,trainProp);
        ShapeletTransformClassifier stMinute=new ShapeletTransformClassifier();
        stMinute.setOneMinuteLimit();
        ShapeletTransformClassifier stHour=new ShapeletTransformClassifier();
        stHour.setOneHourLimit();
//        ShapeletTransformClassifier stDay=new ShapeletTransformClassifier();
//        stDay.setOneDayLimit();
        double acc1=ClassifierTools.singleTrainTestSplitAccuracy(stMinute, split[0], split[1]);
        double acc2=ClassifierTools.singleTrainTestSplitAccuracy(stHour, split[0], split[1]);
//        double acc3=ClassifierTools.singleTrainTestSplitAccuracy(stDay, split[0], split[1]);
        of.writeLine(seriesLength+","+fold);
        of.writeLine(acc1+","+acc2);
    }
    
    public static void generateScripts(boolean grace){
        String path="C:\\Users\\ajb\\Dropbox\\Code\\Cluster Scripts\\SimulatorScripts\\ShapeSims\\";
        OutFile of2=new OutFile(path+"ShapeletLengths.txt");
        for(int length=100;length<=1000;length+=100){    
            OutFile of;    
            if(grace)
                of = new OutFile(path+"ShapeletSimLength"+length+".bsub");
            else
                of = new OutFile(path+"ShapeletSimLength"+length+".bsub");
            of.writeLine("#!/bin/csh");
            if(grace)
                of.writeLine("#BSUB -q long");
            else    
                of.writeLine("#BSUB -q long-eth");
            of.writeLine("#BSUB -J "+"ShapeSim"+length+"[1-100]");
            of.writeLine("#BSUB -oo output/"+"ShapeSim"+length+".out");
            of.writeLine("#BSUB -eo error/"+"ShapeSim"+length+".err");
            of.writeLine("#BSUB -R \"rusage[mem=7000]\"");
            of.writeLine("#BSUB -M 8000");
            if(grace)
                of.writeLine("module add java/jdk/1.8.0_31");
            else
                of.writeLine("module add java/jdk1.8.0_51");
            of.writeLine("java -jar Simulator.jar "+length+" $LSB_JOBINDEX");                

            of2.writeLine("bsub < Scripts/SimulatorExperiments/"+"ShapeletSimLength"+length+".bsub");
        }         
    }
    public static void main(String[] args){
//        generateScripts(false);
//        System.exit(0);
        
        if(args.length>0){
//Set your cluster path gere        
            DataSets.clusterPath="/gpfs/home/ajb/";
//Set wherever you are putting the files here
            DataSets.resultsPath=DataSets.clusterPath+"Results/SimulationExperiments/";
//Arg 1 is series length, Arg 2 is the fold            
            int length=Integer.parseInt(args[0]);
            int fold=Integer.parseInt(args[1])-1;
            shapeletSimulatorWithLength(length,fold);
        }
        else{//Local run for debugging
            DataSets.dropboxPath="C:/Users/ajb/Dropbox/";
            DataSets.resultsPath=DataSets.dropboxPath+"Results/SimulationExperiments/";
            int length=100;
            int fold=0;
            shapeletSimulatorWithLength(length,fold);
        }
    }        
    
    
}
