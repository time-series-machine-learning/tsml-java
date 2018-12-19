/*
1. RISE Tests:
    Test RISE vs PS_Ensemble and ACF_Ensemble on ARSim data

*/
package development.new_COTE_experiments;

import development.DataSets;
import fileIO.OutFile;
import java.util.ArrayList;
import statistics.distributions.Distribution;
import statistics.simulators.DataSimulator;
import statistics.simulators.Model;
import statistics.simulators.SimulateSpectralData;
import statistics.simulators.WhiteNoiseModel;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.core.Instances;
import development.MatrixProfileExperiments;
import static development.MatrixProfileExperiments.setClassifier;
import development.SimulationExperiments;

/**
 *
 * @author ajb
 */
public class RISESimulatedDataExperiments {
    static String resultsPath="";
    static int []casesPerClass={500,500};
    static int seriesLength=500;
    public static double simulatorExperiment(String dataSimulator, String classifier, int fold){
        Classifier c=SimulationExperiments.setClassifier(classifier);
        Instances test= SimulationExperiments.simulateData(dataSimulator,fold);
        System.out.println("Classifier ="+classifier+" Simulator ="+dataSimulator);
        System.out.println("Fold "+fold+" DATA nos atts ="+test.numAttributes()+" noscases ="+test.numInstances());
        double acc=ClassifierTools.stratifiedCrossValidation(test, c, 2, fold);
        System.out.println("Fold "+fold+" classifier acc = "+acc);
        return acc;
    }
 
    public static void RISEEmbeddedModel(boolean hesca) throws Exception{
        //Create relevant classifiers
        int nosClassifiers=7;
        double[] accRecord=new double[nosClassifiers];
        Classifier[] cls=new Classifier[nosClassifiers];
        OutFile out;
        if(hesca)
            out = new OutFile("C:/Temp/RiseHescaTestEmbedded.csv");
        else
            out = new OutFile("C:/Temp/RiseTestEmbedded.csv");
        out.writeLine("100,20,20,100");
        out.writeLine("RotF,DTW,ACF,PS,PSACF,RISE");
 
        for(int i=0;i<100;i++){
            nosClassifiers=0;
            
            cls[nosClassifiers++]=setClassifier("RotF");
            cls[nosClassifiers++]=setClassifier("DTW");
            cls[nosClassifiers++]=setClassifier("AR");
            cls[nosClassifiers++]=setClassifier("PS");
            cls[nosClassifiers++]=setClassifier("PSACF");
           cls[nosClassifiers++]=setClassifier("RISE");
           if(hesca)
               cls[nosClassifiers++]=setClassifier("RISE_HESCA");
            DataSimulator ds=new SimulateSpectralData();
            ds.setCasesPerClass(new int[]{20,20});
            int arLength=100,fullLength=200;
            ds.setLength(arLength);
            Instances[] data=ds.generateTrainTest();
            ArrayList<Model> noise=new ArrayList<>();
            WhiteNoiseModel wm=new WhiteNoiseModel();
            noise.add(wm);
            wm=new WhiteNoiseModel();
            noise.add(wm);
            System.out.println("NOISE SIZE ="+noise.size());
            DataSimulator ds2=new DataSimulator(noise); // By default it goes to white noise 
            ds2.setCasesPerClass(new int[]{20,20});
            ds2.setLength(fullLength);
            Instances[] noiseData=ds2.generateTrainTest();
//Choose random start
            int startPos=(int)(Math.random()*(fullLength-arLength));
            for(int j=startPos;j<startPos+arLength;j++){
                for(int k=0;k<data[0].numInstances();k++)
                    noiseData[0].instance(k).setValue(j, data[0].instance(k).value(j-startPos));
                for(int k=0;k<data[1].numInstances();k++)
                    noiseData[1].instance(k).setValue(j, data[1].instance(k).value(j-startPos));
            }
            
/*             OutFile of =new OutFile("C:/Temp/train"+i+".csv");
             of.writeLine(data[0].toString());
            of =new OutFile("C:/Temp/test"+i+".csv");
             of.writeLine(data[1].toString());             
 */         System.out.println("Data Generated for fold "+i);
            for(int j=0;j<nosClassifiers;j++){
                System.out.println("Training "+cls[j].getClass().getName());
                double acc=ClassifierTools.singleTrainTestSplitAccuracy(cls[j], noiseData[0],noiseData[1]);
                accRecord[j]+=acc;
                out.writeString(acc+",");
                System.out.println(" accuracy ="+acc+" mean = "+(accRecord[j]/(i+1)));
            }
            out.writeString("\n");
        }
    }
    
    
    public static void RISEFullModel(boolean hesca) throws Exception{
        //Create relevant classifiers
        int nosClassifiers=7;
        double[] accRecord=new double[nosClassifiers];
        Classifier[] cls=new Classifier[nosClassifiers];
        OutFile out;
        if(hesca)
            out = new OutFile("C:/Temp/RiseHescaTest.csv");
        else
            out = new OutFile("C:/Temp/RiseTest.csv");
        out.writeLine("100,20,20,100");
        out.writeLine("RotF,DTW,ACF,PS,PSACF,RISE");
 
        for(int i=0;i<100;i++){
            nosClassifiers=0;
            
            cls[nosClassifiers++]=setClassifier("RotF");
            cls[nosClassifiers++]=setClassifier("DTW");
            cls[nosClassifiers++]=setClassifier("AR");
            cls[nosClassifiers++]=setClassifier("PS");
            cls[nosClassifiers++]=setClassifier("PSACF");
           cls[nosClassifiers++]=setClassifier("RISE");
           if(hesca)
               cls[nosClassifiers++]=setClassifier("RISE_HESCA");
            DataSimulator ds=new SimulateSpectralData();
            ds.setCasesPerClass(new int[]{20,20});
            ds.setLength(100);
            Instances[] data=ds.generateTrainTest();
/*             OutFile of =new OutFile("C:/Temp/train"+i+".csv");
             of.writeLine(data[0].toString());
            of =new OutFile("C:/Temp/test"+i+".csv");
             of.writeLine(data[1].toString());             
 */         System.out.println("Data Generated for fold "+i);
            for(int j=0;j<nosClassifiers;j++){
                System.out.println("Training "+cls[j].getClass().getName());
                double acc=ClassifierTools.singleTrainTestSplitAccuracy(cls[j], data[0],data[1]);
                accRecord[j]+=acc;
                out.writeString(acc+",");
                System.out.println(" accuracy ="+acc+" mean = "+(accRecord[j]/(i+1)));
            }
            out.writeString("\n");
        }
    }
    public static void RISE_Cluster(String foldStr) throws Exception{
        //Create relevant classifiers
        OutFile out=new OutFile(resultsPath+"Simulation1_Fold"+foldStr+".csv");
        int trainSize=40;
        int testSize=40;
        int length=100;
        int fold=Integer.parseInt(foldStr)-1;
        out.writeLine("fold,trainSize,testSize,length");
        out.writeLine(fold+","+trainSize+","+testSize+","+length);
        out.writeLine("RotF,DTW,ACF,PS,PSACF,RISE");
        int nosClassifiers=6;
        Classifier[] cls=new Classifier[nosClassifiers];
        nosClassifiers=0;
        cls[nosClassifiers++]=setClassifier("RotF");
        cls[nosClassifiers++]=setClassifier("DTW");
        cls[nosClassifiers++]=setClassifier("AR");
        cls[nosClassifiers++]=setClassifier("PS");
        cls[nosClassifiers++]=setClassifier("PSACF");
        cls[nosClassifiers++]=setClassifier("RISE");
        DataSimulator ds=new SimulateSpectralData();
        Distribution.setDistributionSeed(fold);
        ds.setCasesPerClass(new int[]{trainSize/2,trainSize/2});
        ds.setLength(length);
        Instances[] data=ds.generateTrainTest();
        for(int j=0;j<nosClassifiers;j++){
            double acc=ClassifierTools.singleTrainTestSplitAccuracy(cls[j], data[0],data[1]);
            out.writeString(acc+",");
        }
        out.writeString("\n");
    }
    
    
    public static void main(String[] args) throws Exception {
        
        if(args.length>0){
            resultsPath=DataSets.clusterPath+"SimulationResults/";
            RISE_Cluster(args[0]);
        }
        else{
            RISEEmbeddedModel(false);            
//            RISEFullModel(false);
        }
    }
//Does the resampling make a deep copy or not?    
    public static void testLabels(){
        Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\Coffee\\Coffee_TRAIN");
        Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\Coffee\\Coffee_TEST");
        Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, 0);
        double[] testLabels=new double[data[1].numInstances()];
        for(int i=0;i<testLabels.length;i++){
            testLabels[i]=data[1].instance(i).classValue();
            data[1].instance(i).setClassMissing();
            System.out.println("Label:="+testLabels[i]+" is it missing? "+data[1].instance(i).classIsMissing()+" actual value ="+data[1].instance(i).value(data[1].numAttributes()-1));
        }
        for(int i=0;i<train.numInstances();i++)
            System.out.println("TRAIN: is it missing? "+train.instance(i).classIsMissing()+" actual value ="+train.instance(i).value(data[1].numAttributes()-1));
        for(int i=0;i<test.numInstances();i++)
            System.out.println("TEST: is it missing? "+test.instance(i).classIsMissing()+" actual value ="+test.instance(i).value(data[1].numAttributes()-1));

    
    }
    
}
