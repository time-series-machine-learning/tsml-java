/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package tsml.examples;

import tsml.classifiers.distance_based.DTWCV;
import tsml.classifiers.legacy.COTE.FlatCote;
import tsml.classifiers.shapelet_based.LearnShapelets;
import tsml.classifiers.shapelet_based.FastShapelets;
import tsml.classifiers.interval_based.TSBF;
import tsml.classifiers.interval_based.TSF;
import tsml.classifiers.distance_based.DTD_C;
import tsml.classifiers.dictionary_based.BOSS;
import tsml.classifiers.legacy.RISE;
import tsml.classifiers.shapelet_based.ShapeletTransformClassifier;
import tsml.classifiers.interval_based.LPS;
import tsml.classifiers.distance_based.ElasticEnsemble;
import tsml.classifiers.distance_based.DD_DTW;
import tsml.classifiers.dictionary_based.BagOfPatternsClassifier;
import tsml.classifiers.legacy.COTE.HiveCote;
import fileIO.OutFile;
import statistics.simulators.Model;
import statistics.simulators.SimulateSpectralData;
import statistics.simulators.SimulateDictionaryData;
import statistics.simulators.SimulateIntervalData;
import statistics.simulators.SimulateShapeletData;
import tsml.classifiers.EnhancedAbstractClassifier;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.meta.RotationForest;
import machine_learning.classifiers.ensembles.CAWPE;
import machine_learning.classifiers.ensembles.SaveableEnsemble;
import machine_learning.classifiers.tuned.TunedRandomForest;
import weka.core.Instances;
import utilities.ClassifierTools;

/**
 * 
 * @author ajb
 * FINAL VERSION of simulator experiments for stand alone execution only
 * Just the main experiments, copied here for clarity. For sensitivity analysis
 * and cluster based versions, see the class 
 * Please read the technical report 
LINK HERE
 */
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
                c=new DTWCV();
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
                c=new BagOfPatternsClassifier();
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
//                ((RISE)c).setTransformType(RISE.TransformType.ACF_PS);
                ((RISE)c).setNumClassifiers(500);
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
        if(EnhancedAbstractClassifier.classifierAbleToEstimateOwnPerformance(c))
            ((EnhancedAbstractClassifier)c).setEstimateOwnPerformance(true);
        if(c instanceof SaveableEnsemble)
           ((SaveableEnsemble)c).saveResults(preds+"/internalCV_"+sample+".csv",preds+"/internalTestPreds_"+sample+".csv");
        try{              
            c.buildClassifier(train);
            if(EnhancedAbstractClassifier.classifierIsEstimatingOwnPerformance(c))
                ((EnhancedAbstractClassifier)c).getTrainResults().writeFullResultsToFile(preds+"/trainFold"+sample+".csv");
            
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
            if(c instanceof EnhancedAbstractClassifier)
                p.writeLine(((EnhancedAbstractClassifier)c).getParameters());
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
                        c=new DTWCV();
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
                        c=new BagOfPatternsClassifier();
                        break;
                    case "BOSS":
                        c=new BOSS();
                        break;
                    case "COTE":
                        c=new FlatCote();
                        break;
                    case "RISE":
                        c=new RISE();
//                        ((RISE)c).setTransformType(RISE.TransformType.ACF_PS);
                        ((RISE)c).setNumClassifiers(500);
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
        runShapeletSimulatorExperiment();
    }
}
    
