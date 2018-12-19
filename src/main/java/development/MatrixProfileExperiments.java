/*
Class to run one of various simulations.  
*/
package development;

import timeseriesweka.classifiers.FlatCote;
import timeseriesweka.classifiers.LearnShapelets;
import timeseriesweka.classifiers.FastShapelets;
import timeseriesweka.classifiers.TSBF;
import timeseriesweka.classifiers.TSF;
import timeseriesweka.classifiers.DTD_C;
import timeseriesweka.classifiers.BOSS;
import timeseriesweka.classifiers.RISE;
import timeseriesweka.classifiers.ShapeletTransformClassifier;
import timeseriesweka.classifiers.LPS;
import timeseriesweka.classifiers.ElasticEnsemble;
import timeseriesweka.classifiers.DD_DTW;
import timeseriesweka.classifiers.BagOfPatterns;
import timeseriesweka.classifiers.HiveCote;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import java.util.logging.Level;
import java.util.logging.Logger;
import statistics.simulators.DataSimulator;
import statistics.simulators.ElasticModel;
import statistics.simulators.Model;
import statistics.simulators.SimulateSpectralData;
import statistics.simulators.SimulateDictionaryData;
import statistics.simulators.SimulateIntervalData;
import statistics.simulators.SimulateShapeletData;
import statistics.simulators.SimulateWholeSeriesData;
import statistics.simulators.SimulateElasticData;
import statistics.simulators.SimulateMatrixProfileData;
import static statistics.simulators.SimulateMatrixProfileData.generateMatrixProfileData;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import weka.classifiers.Classifier;
import timeseriesweka.classifiers.FastDTW_1NN;
import weka.classifiers.meta.RotationForest;
import vector_classifiers.CAWPE;
import timeseriesweka.classifiers.ensembles.SaveableEnsemble;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.DTW1NN;
import timeseriesweka.filters.MatrixProfile;
import vector_classifiers.TunedRandomForest;
import weka.core.Instances;
import utilities.ClassifierTools;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.lazy.kNN;
import weka.filters.NormalizeCase;
/*
AJB Oct 2016

Model to simulate data where matrix profile should be optimal.

/*
Basic experimental design in SimulationExperiments.java:
Simulate data, [possibly normalise for standard classifiers], 
take MP build ED on that. 

Variants tried are in 
First scenario:
Each class is defined by two locations, noise is very low (0.1)
Config 1: 
Cycle through the two possible shapes, give random base and amplitude to both
Model calls : ((MatrixProfileModel)model).

No normalisation: MatrixProfileExperiments.normalize=false;
ED mean acc =0.7633
MP_ED mean acc =1
DTW mean acc =0.6389
RotF mean acc =0.5444
ST mean acc =0.6333
TSF mean acc =1
BOSS mean acc =0.7844

Normalisation: MatrixProfileExperiments.normalize=true
ED mean acc =1
MP_ED mean acc =1
DTW mean acc =0.6567
RotF mean acc =0.7522
ST mean acc =1
TSF mean acc =0.5767
BOSS mean acc =1

Unfortunately need to normalise for any credibility, so on to cnfig 2:

Config 2: Give different random base and amplitude to both
No normalisation: (ran these by mistake, will no longer run normalisation)
ED mean acc =0.7822
MP_ED mean acc =0.9844
DTW mean acc =0.6144
RotF mean acc =0.539
ST mean acc =0.57777
TSF mean acc =1.0
BOSS mean acc =0.773

Normalisation:
ED mean acc =1
MP_ED mean acc =0.9844
DTW mean acc =0.6467
RotF mean acc =0.8133
ST mean acc =1
TSF mean acc =0.6856
BOSS mean acc =1

Config 3:
After shock model. 
1. Make second shape smaller than the first. 
2. Fix position of first shape. 
2. Make one model have only one shape.

WAIT and go back. Set up with amplitude between 2 and 4 we get this. 
 Sig =0.1 Mean 1NN Acc =0.81222 Mean 1NN Norm Acc =0.91889 Mean 1NN MP Acc = 1

CHANGE: Ranomise the shape completely!


*/
public class MatrixProfileExperiments {
    static boolean local=false;
    static int []casesPerClass={50,50};
    static int seriesLength=500;
    static double trainProp=0.5;
    static boolean normalize=true;
    static String[] allClassifiers={ //Benchmarks
        "ED", "RotF","DTW",
        //Whole series
//        "DD_DTW","DTD_C",
        "EE","HESCA",
        //Interval
        "TSF",
//        "TSBF","LPS",
        //Shapelet
//        "FastShapelets","LearnShapelets",
        "ST",
        //Dictionary        "BOP",
        "BOSS",
        //Spectral
        "RISE",
        //Combos
        "FLATCOTE","HIVECOTE"};
    
    
    public static Classifier setClassifier(String str) throws RuntimeException{
        
        Classifier c;
        switch(str){
            case "ED": case "MP_ED":
                c=new kNN(1);
                break;
            case "HESCA":
                c=new CAWPE();
                break;
            case "RandF": case "MP_RotF":
                c=new TunedRandomForest();
                break;
            case "RotF":
                c=new RotationForest();
                break;
            case "DTW": case "MP_DTW":
                c=new DTW1NN();
                break;
             case "DD_DTW":
                c=new DD_DTW();
                break;               
            case "DTD_C":    
                c=new DTD_C();
                break;               
            case "EE":    
                c=new ElasticEnsemble();
                break;                          
            case "TSF":
                c=new TSF();
                break;
            case "TSBF":
                c=new TSBF();
                break;
            case "LPS":
                c=new LPS();
                break;
            case "FastShapelets":
                c=new FastShapelets();
                break;
            case "ST":
                c=new ShapeletTransformClassifier();
                if(local)
                    ((ShapeletTransformClassifier)c).setOneMinuteLimit();
                else
                   ((ShapeletTransformClassifier)c).setOneHourLimit();
//                ((ShapeletTransformClassifier)c).setOneMinuteLimit();//DEBUG
                break;
            case "BOP":
                c=new BagOfPatterns();
                break;
            case "BOSS":
                c=new BOSS();
                break;
            case "COTE":
            case "FLATCOTE":
                c=new FlatCote();
                break;
            case "HIVECOTE":
                c=new HiveCote();
//                ((HiveCote)c).setNosHours(2);
                break;
            case "RISE":
                c=new RISE();
                ((RISE)c).setTransformType("PS_ACF");
                ((RISE)c).setNosBaseClassifiers(500);
                break;
            case "RISE_HESCA":
                c=new RISE();
                ((RISE)c).setTransformType("PS_ACF");
                Classifier base=new CAWPE();
                ((RISE)c).setBaseClassifier(base);
                ((RISE)c).setNosBaseClassifiers(20);
                break;
            default:
                throw new RuntimeException(" UNKNOWN CLASSIFIER "+str);
        }
        return c;
    }
    
    
   

    public static void main(String[] args) throws Exception{
        seriesLength=200;
        trainProp=0.1;
        casesPerClass=new int[]{50,50};
        Model.setDefaultSigma(1);
        int folds=100;
        int numMPClassifiers=1;
        String[] algos={"MP_ED","ED","RotF","DTW","ST","TSF","BOSS","RISE","COTE"};
//,,"MP_RotF","MP_DTW"};
        double[] means=new double[algos.length];
        OutFile mpExample=new OutFile("C:\\temp\\mpResults.csv"); 
        OutFile mpW=new OutFile("C:\\temp\\mpWindows.csv"); 
        for(int j=0;j<algos.length;j++)
            mpExample.writeString(","+algos[j]);
        mpExample.writeString("\n");
        
//Generate data        
        for(int i=0;i<folds;i++){
            mpExample.writeString((i+1)+"");
            Model.setGlobalRandomSeed(i);
            Instances d=SimulateMatrixProfileData.generateMatrixProfileData(seriesLength,casesPerClass);
            Instances[] split=InstanceTools.resampleInstances(d,i,trainProp);
            kNN knn= new kNN();
            knn.setKNN(1);
            MatrixProfile mp=new MatrixProfile(29);
            Instances[] mpSplit=new Instances[2];
            Instances[] normSplit=new Instances[2];
            mpSplit[0]=mp.process(split[0]);
            mpSplit[1]=mp.process(split[1]);
            NormalizeCase nc=new NormalizeCase();
            normSplit[0]=nc.process(split[0]);
            normSplit[1]=nc.process(split[1]);
            for(int j=0;j<algos.length;j++){

                Classifier c=setClassifier(algos[j]);
                double acc;
                if(algos[j].contains("MP_"))
                    acc=ClassifierTools.singleTrainTestSplitAccuracy(c, mpSplit[0], mpSplit[1]);
                else
                    acc=ClassifierTools.singleTrainTestSplitAccuracy(c, normSplit[0], normSplit[1]);
                System.out.println("Classifier "+algos[j]+" acc ="+acc);
                means[j]+=acc;
                mpExample.writeString(","+(1-acc));
            }
                mpExample.writeString("\n");
        }
        for(int j=0;j<algos.length;j++)
            System.out.println(algos[j]+" mean acc = "+(means[j]/folds));
            
        
        
/*
        if(args.length>0){
            DataSets.resultsPath=DataSets.clusterPath+"Results/SimulationExperiments/";
            if(args.length==3){//Base experiment
                double b=runSimulationExperiment(args,true);
                System.out.println(args[0]+","+args[1]+","+","+args[2]+" Acc ="+b);
            }else if(args.length==4){//Error experiment)
                runErrorExperiment(args);
                
            }
//              runLengthExperiment(paras);
        }
        else{
//            DataSets.resultsPath="C:\\Users\\ajb\\Dropbox\\Results\\MatrixProfileExperiments\\";
            local=true;
            DataSets.resultsPath="C:\\temp\\";
                String[] algos={"ED","MP_ED"};//,,"MP_RotF","MP_DTW"};
                double[] meanAcc=new double[algos.length];
            for(int i=1;i<=10;i++){
                for(int j=0;j<algos.length;j++){
                    String[] para={"MatrixProfile",algos[j],i+""};
                    double b=runSimulationExperiment(para,true);
                    meanAcc[j]+=b;
                    System.out.println(para[0]+","+para[1]+","+","+para[2]+" Acc ="+b);
                    
                }
            } 
            DecimalFormat df=new DecimalFormat("##.####");
            for(int j=0;j<algos.length;j++)
                System.out.println(algos[j]+" mean acc ="+df.format(meanAcc[j]/10));
        }
*/
    }
}
