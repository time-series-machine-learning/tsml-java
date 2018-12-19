/**
 *
 * @author ajb
 * FINAL VERSION of simulator experiments for stand alone execution only
 * Just the main experiments, copied here for clarity. For sensitivity analysis
 * and cluster based versions, see the class 
 * Please read the technical report 
LINK HERE
*/
package timeseriesweka.examples;

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
import development.DataSets;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import statistics.simulators.DataSimulator;
import statistics.simulators.Model;
import statistics.simulators.SimulateSpectralData;
import statistics.simulators.SimulateDictionaryData;
import statistics.simulators.SimulateIntervalData;
import statistics.simulators.SimulateShapeletData;
import statistics.simulators.SimulateWholeSeriesData;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import weka.classifiers.Classifier;
import timeseriesweka.classifiers.FastDTW_1NN;
import weka.classifiers.meta.RotationForest;
import vector_classifiers.CAWPE;
import timeseriesweka.classifiers.ensembles.SaveableEnsemble;
import vector_classifiers.TunedRandomForest;
import weka.core.Instances;
import utilities.ClassifierTools;
import utilities.TrainAccuracyEstimate;

public class SimulationExperiments {
//Global variables that relate to the data set. These are different for different
//simulators, and are set to default values in setStandardGlobalParameters    
    static int []casesPerClass={50,50};
    static int seriesLength=300;
    static double trainProp=0.5;
 
//<editor-fold defaultstate="collapsed" desc="All Classifiers: edit if you want to try some others">     
    static String[] allClassifiers={ //Benchmarks
        "RotF","DTW","HESCA",
        //Whole series
        "DD_DTW","DTD_C","EE","HESCA",
        //Interval
        "TSF","TSBF","LPS",
        //Shapelet
        "FastShapelets","ST","LearnShapelets",
        //Dictionary
        "BOP","BOSS",
        //Spectral
        "RISE",
        //Combos
        "COTE","FLATCOTE","HIVECOTE"};
      //</editor-fold>     
    
    
 //<editor-fold defaultstate="collapsed" desc="All Simulators: ">    
    static String[] allSimulators={"WholeSeries","Interval","Shapelet","Dictionary","ARMA"};
      //</editor-fold>     
    
    
    public static Classifier createClassifier(String str) throws RuntimeException{
        Classifier c;
        switch(str){
            case "HESCA":
                c=new CAWPE();
                break;
            case "RandF":
                c=new TunedRandomForest();
                break;
            case "RotF":
                c=new RotationForest();
                break;
            case "DTW":
                c=new FastDTW_1NN();
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
//Just to make sure it is feasible                
               ((ShapeletTransformClassifier)c).setOneHourLimit();
                break;
            case "LearnShapelets":
                c=new LearnShapelets();
                ((LearnShapelets)c).setParamSearch(true);
                break;
            case "BOP":
                c=new BagOfPatterns();
                break;
            case "BOSS":
                c=new BOSS();
                break;
            case "FLATCOTE":
                c=new FlatCote();
                break;
            case "HIVECOTE":
                c=new HiveCote();
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
    
    public static void setStandardGlobalParameters(String str){
         switch(str){
            case "ARMA": case "AR": case "Spectral":
                casesPerClass=new int[]{200,200};
                seriesLength=200;
                trainProp=0.1;
                Model.setDefaultSigma(1);
                break;
            case "Shapelet": 
                casesPerClass=new int[]{250,250};
                seriesLength=300;
                trainProp=0.1;
                Model.setDefaultSigma(1);
                break;
            case "Dictionary":
                casesPerClass=new int[]{200,200};
                seriesLength=1500;
                trainProp=0.1;
                SimulateDictionaryData.setShapeletsPerClass(new int[]{5,10});
                SimulateDictionaryData.setShapeletLength(29);
 //               SimulateDictionaryData.checkGlobalSeedForIntervals();
                Model.setDefaultSigma(1);
               break; 
            case "Interval":
                seriesLength=1000;
                trainProp=0.1;
                casesPerClass=new int[]{200,200};
                Model.setDefaultSigma(1);
//                SimulateIntervalData.setAmp(1);
                SimulateIntervalData.setNosIntervals(3);
                SimulateIntervalData.setNoiseToSignal(10);
                break;
           case "WholeSeriesElastic":
                seriesLength=200;
                trainProp=0.1;
                casesPerClass=new int[]{200,200};
                Model.setDefaultSigma(1);
 //               SimulateWholeSeriesElastic.
                break;
            case "WholeSeries":
//                break;
        default:
                throw new RuntimeException(" UNKNOWN SIMULATOR ");
            
        }       
    }
    
 /**
  * Creates a simulated data set with the data characteristics defined in this 
  * class and the default model characteristics set in the appropriate Simulator class
  * If you want to control the model parameters, see the class DataSimulator for two alternative
  * use cases
  * @param str:name of the simulator
  * @param seed: random seed
  * @return
  * @throws RuntimeException 
  */   
    public static Instances simulateData(String str,int seed) throws RuntimeException{
        Instances data;
//        for(int:)
        Model.setGlobalRandomSeed(seed);
        switch(str){
            case "ARMA": case "AR":
                 data=SimulateSpectralData.generateARDataSet(seriesLength, casesPerClass, true);
                break;
            case "Shapelet": 
                data=SimulateShapeletData.generateShapeletData(seriesLength,casesPerClass);
                break;
            case "Dictionary":
                data=SimulateDictionaryData.generateDictionaryData(seriesLength,casesPerClass);
               break; 
            case "WholeSeries":
 //               data=SimulateWholeSeriesData.generateWholeSeriesData(seriesLength,casesPerClass);
//                break;
           case "WholeSeriesElastic":
 //               data=SimulateWholeSeriesData.generateWholeSeriesData(seriesLength,casesPerClass);
//                break;
        default:
                throw new RuntimeException(" UNKNOWN SIMULATOR ");
            
        }
        return data;
    }
    
/** Runs a single fold experiment, saving all output. 
 * 
 * @param train
 * @param test
 * @param c
 * @param sample
 * @param preds
 * @return 
 */
    public static double singleSampleExperiment(Instances train, Instances test, Classifier c, int sample,String preds){
        double acc=0;
        OutFile p=new OutFile(preds+"/testFold"+sample+".csv");

// hack here to save internal CV for further ensembling   
        if(c instanceof TrainAccuracyEstimate)
            ((TrainAccuracyEstimate)c).writeCVTrainToFile(preds+"/trainFold"+sample+".csv");
        if(c instanceof SaveableEnsemble)
           ((SaveableEnsemble)c).saveResults(preds+"/internalCV_"+sample+".csv",preds+"/internalTestPreds_"+sample+".csv");
        try{              
            c.buildClassifier(train);
            int[][] predictions=new int[test.numInstances()][2];
            for(int j=0;j<test.numInstances();j++){
                predictions[j][0]=(int)test.instance(j).classValue();
                test.instance(j).setMissing(test.classIndex());//Just in case ....
            }
            for(int j=0;j<test.numInstances();j++)
            {
                predictions[j][1]=(int)c.classifyInstance(test.instance(j));
                if(predictions[j][0]==predictions[j][1])
                    acc++;
            }
            acc/=test.numInstances();
            String[] names=preds.split("/");
            p.writeLine(names[names.length-1]+","+c.getClass().getName()+",test");
            if(c instanceof SaveParameterInfo)
                p.writeLine(((SaveParameterInfo)c).getParameters());
            else if(c instanceof SaveableEnsemble)
                p.writeLine(((SaveableEnsemble)c).getParameters());
            else
                p.writeLine("NoParameterInfo");
            p.writeLine(acc+"");
            for(int j=0;j<test.numInstances();j++){
                p.writeString(predictions[j][0]+","+predictions[j][1]+",");
                double[] dist =c.distributionForInstance(test.instance(j));
                for(double d:dist)
                    p.writeString(","+d);
                p.writeString("\n");
            }
        }catch(Exception e)
        {
                System.out.println(" Error ="+e+" in method simpleExperiment"+e);
                e.printStackTrace();
                System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
                System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes and "+test.numInstances()+" instances");

                System.exit(0);
        }
         return acc;
    }

  
/** 
 * 
 * Stand alone method to exactly reproduce shapelet experiment for all 
 * classifiers defined within this method (makes NO use of global variables defined above.
 */    
    public static void runShapeletSimulatorExperiment(){
        Model.setDefaultSigma(1);
        seriesLength=300;
        casesPerClass=new int[]{50,50};
        String[] classifiers={"RotF","DTW","FastShapelets","ST","BOSS"};
//            "EE","CAWPE","TSF","TSBF","FastShapelets","ST","LearnShapelets","BOP","BOSS","RISE","COTE"};
        OutFile of=new OutFile("C:\\Temp\\ShapeletSimExperiment.csv");
        setStandardGlobalParameters("Shapelet");
        of.writeLine("Shapelet Sim, series length= "+seriesLength+" cases class 0 ="+casesPerClass[0]+" class 1"+casesPerClass[0]+" train proportion = "+trainProp);
        of.writeString("Rep");
        for(String s:classifiers)
            of.writeString(","+s);
        of.writeString("\n");
        for(int i=0;i<100;i++){
            of.writeString(i+",");
//Generate data
            Model.setGlobalRandomSeed(i);
            Instances data=SimulateShapeletData.generateShapeletData(seriesLength,casesPerClass);
//Split data
            Instances[] split=InstanceTools.resampleInstances(data, i,trainProp);
            for(String str:classifiers){
                Classifier c;
        //Build classifiers            
                switch(str){
                    case "RotF":
                        c=new RotationForest();
                        break;
                    case "DTW":
                        c=new FastDTW_1NN();
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
                    case "FastShapelets":
                        c=new FastShapelets();
                        break;
                    case "ST":
                        c=new ShapeletTransformClassifier();
                            ((ShapeletTransformClassifier)c).setOneHourLimit();
                        break;
                    case "LearnShapelets":
                        c=new LearnShapelets();
                        break;
                    case "BOP":
                        c=new BagOfPatterns();
                        break;
                    case "BOSS":
                        c=new BOSS();
                        break;
                    case "COTE":
                        c=new FlatCote();
                        break;
                    case "RISE":
                        c=new RISE();
                        ((RISE)c).setTransformType("PS_ACF");
                        ((RISE)c).setNosBaseClassifiers(500);
                        break;
                    default:
                        throw new RuntimeException(" UNKNOWN CLASSIFIER "+str);
                }
                double acc=ClassifierTools.singleTrainTestSplitAccuracy(c, split[0], split[1]);
                of.writeString(acc+",");
                System.out.println(i+" "+str+" acc ="+acc);
            }
            of.writeString("\n");
        }
        
    }


    
    public static void main(String[] args){
        DataSets.resultsPath="C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\";
        runShapeletSimulatorExperiment();
    }
}
    
