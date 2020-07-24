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
package experiments;

//import com.sun.management.GarbageCollectionNotificationInfo;
//import com.sun.management.GarbageCollectorMXBean;
import tsml.classifiers.dictionary_based.*;
import tsml.classifiers.distance_based.DTWCV;
import tsml.classifiers.legacy.COTE.FlatCote;
import tsml.classifiers.shapelet_based.LearnShapelets;
import tsml.classifiers.shapelet_based.FastShapelets;
import tsml.classifiers.interval_based.TSBF;
import tsml.classifiers.interval_based.TSF;
import tsml.classifiers.distance_based.DTD_C;
import tsml.classifiers.shapelet_based.ShapeletTransformClassifier;
import tsml.classifiers.interval_based.LPS;
import tsml.classifiers.distance_based.ElasticEnsemble;
import tsml.classifiers.distance_based.DD_DTW;
import tsml.classifiers.legacy.COTE.HiveCote;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

import statistics.simulators.ElasticModel;
import statistics.simulators.Model;
import statistics.simulators.SimulateSpectralData;
import statistics.simulators.SimulateDictionaryData;
import statistics.simulators.SimulateIntervalData;
import statistics.simulators.SimulateShapeletData;
import statistics.simulators.SimulateWholeSeriesData;
import statistics.simulators.SimulateElasticData;
import statistics.simulators.SimulateMatrixProfileData;
import tsml.classifiers.EnhancedAbstractClassifier;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.meta.RotationForest;
import machine_learning.classifiers.ensembles.CAWPE;
import machine_learning.classifiers.ensembles.SaveableEnsemble;
import tsml.classifiers.legacy.elastic_ensemble.DTW1NN;
import tsml.transformers.MatrixProfile;
import weka.core.Instances;
import utilities.ClassifierTools;
import machine_learning.classifiers.kNN;
import weka.core.Instance;
import tsml.transformers.RowNormalizer;

import javax.management.Notification;

/*

Class to run one of various simulations.  

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
public class SimulationExperiments {
    static boolean local = false;
    static int[] casesPerClass = { 50, 50 };
    static int seriesLength = 500;
    static double trainProp = 0.5;
    static boolean normalize = true;
    static String[] allClassifiers = { // Benchmarks
            "ED", "RotF", "DTW",
            // Whole series
            // "DD_DTW","DTD_C",
            "EE", "HESCA",
            // Interval
            "TSF",
            // "TSBF","LPS",
            // Shapelet
            // "FastShapelets","LearnShapelets",
            "ST",
            // Dictionary "BOP",
            "BOSS",
            // Spectral
            "RISE",
            // Combos
            "FLATCOTE", "HIVECOTE" };
    static String[] allSimulators = { "WholeSeriesElastic", "Interval", "Shapelet", "Dictionary", "ARMA" };

    public static Classifier setClassifier(String str) throws RuntimeException {

        Classifier c;
        switch (str) {
            case "ED":
            case "MP_ED":
                c = new kNN(1);
                break;
            case "HESCA":
                c = new CAWPE();
                break;
            case "RotF":
                c = new RotationForest();
                break;
            case "DTW":
            case "MP_DTW":
                c = new DTW1NN();
                break;
            case "DD_DTW":
                c = new DD_DTW();
                break;
            case "DTD_C":
                c = new DTD_C();
                break;
            case "EE":
                c = new ElasticEnsemble();
                break;
            case "TSF":
                c = new TSF();
                break;
            case "TSBF":
                c = new TSBF();
                break;
            case "LPS":
                c = new LPS();
                break;
            case "FastShapelets":
                c = new FastShapelets();
                break;
            case "ST":
                c = new ShapeletTransformClassifier();
                if (local)
                    ((ShapeletTransformClassifier) c).setOneMinuteLimit();
                else
                    ((ShapeletTransformClassifier) c).setOneHourLimit();
                // ((ShapeletTransformClassifier)c).setOneMinuteLimit();//DEBUG
                break;
            case "BOP":
                c = new BagOfPatternsClassifier();
                break;
            case "BOSS":
                c = new BOSS();
                break;
            case "COTE":
            case "FLATCOTE":
                c = new FlatCote();
                break;
            case "HIVECOTE":
                c = new HiveCote();
                // ((HiveCote)c).setNosHours(2);
                break;
            default:
                throw new RuntimeException(" UNKNOWN CLASSIFIER " + str);
        }
        return c;
    }

    public static void setStandardGlobalParameters(String str) {
        switch (str) {
            case "ARMA":
            case "AR":
            case "Spectral":
                casesPerClass = new int[] { 200, 200 };
                seriesLength = 200;
                trainProp = 0.1;
                Model.setDefaultSigma(1);
                break;
            case "Shapelet":
                casesPerClass = new int[] { 250, 250 };
                seriesLength = 300;
                trainProp = 0.1;
                Model.setDefaultSigma(1);
                break;
            case "Dictionary":
                casesPerClass = new int[] { 200, 200 };
                seriesLength = 1500;
                trainProp = 0.1;
                SimulateDictionaryData.setShapeletsPerClass(new int[] { 5, 10 });
                SimulateDictionaryData.setShapeletLength(29);
                // SimulateDictionaryData.checkGlobalSeedForIntervals();
                Model.setDefaultSigma(1);
                break;
            case "Interval":
                seriesLength = 1000;
                trainProp = 0.1;
                casesPerClass = new int[] { 200, 200 };
                Model.setDefaultSigma(1);
                // SimulateIntervalData.setAmp(1);
                SimulateIntervalData.setNosIntervals(3);
                SimulateIntervalData.setNoiseToSignal(10);
                break;
            case "WholeSeriesElastic":
            case "WholeSeries":
                seriesLength = 100;
                trainProp = 0.1;
                casesPerClass = new int[] { 100, 100 };
                Model.setDefaultSigma(1);
                ElasticModel.setBaseAndAmp(-2, 4);
                ElasticModel.setWarpPercent(0.4);
                // SimulateWholeSeriesElastic.
                break;
            case "MatrixProfile":
                seriesLength = 150;
                trainProp = 0.1;
                casesPerClass = new int[] { 50, 50 };
                Model.setDefaultSigma(1);
                break;
            default:
                throw new RuntimeException(" UNKNOWN SIMULATOR ");

        }
    }

    public static Instances simulateData(String str, int seed) throws RuntimeException {
        Instances data;
        // for(int:)
        Model.setGlobalRandomSeed(seed);
        switch (str) {
            case "ARMA":
            case "AR":
            case "SPECTRAL":

                data = SimulateSpectralData.generateSpectralEmbeddedData(seriesLength, casesPerClass);
                // data=SimulateSpectralData.generateARDataSet(seriesLength, casesPerClass,
                // true);
                break;
            case "Shapelet":
                data = SimulateShapeletData.generateShapeletData(seriesLength, casesPerClass);
                break;
            case "Dictionary":
                data = SimulateDictionaryData.generateDictionaryData(seriesLength, casesPerClass);
                break;
            case "Interval":
                data = SimulateIntervalData.generateIntervalData(seriesLength, casesPerClass);
                break;

            case "WholeSeries":
                data = SimulateWholeSeriesData.generateWholeSeriesdData(seriesLength, casesPerClass);
                break;
            case "WholeSeriesElastic":
                data = SimulateElasticData.generateElasticData(seriesLength, casesPerClass);
                break;
            case "MatrixProfile":
                data = SimulateMatrixProfileData.generateMatrixProfileData(seriesLength, casesPerClass);
                break;

            default:
                throw new RuntimeException(" UNKNOWN SIMULATOR " + str);

        }
        return data;
    }

    // arg[0]: simulator
    // arg[1]: classifier
    // arg[2]: fold number
    public static double runSimulationExperiment(String[] args, boolean useStandard) throws Exception {
        String simulator = args[0];
        if (useStandard)
            setStandardGlobalParameters(simulator);
        String classifier = args[1];
        Classifier c = setClassifier(classifier);
        int fold = Integer.parseInt(args[2]) - 1;
        String resultsPath = args[3];

        // Set up the train and test files
        File f = new File(resultsPath + simulator);
        if (!f.exists())
            f.mkdirs();
        String predictions = resultsPath + simulator + "/" + classifier;
        f = new File(predictions);
        if (!f.exists())
            f.mkdir();
        // Check whether fold already exists, if so, dont do it, just quit
        f = new File(predictions + "/testFold" + fold + ".csv");
        if (!f.exists() || f.length() == 0) {
            // Do the experiment: find train preds through cross validation
            // Then generate all test predictions
            Instances data = simulateData(args[0], fold);
            Instances[] split = InstanceTools.resampleInstances(data, fold, trainProp);
            System.out.println(" Train size =" + split[0].numInstances() + " test size =" + split[1].numInstances());
            // Check if it is MP or not
            if (classifier.contains("MP_")) {
                try {
                    System.out.println("MAtrix profile run ....");
                    MatrixProfile mp = new MatrixProfile(29);
                    split[0] = mp.transform(split[0]);
                    split[1] = mp.transform(split[1]);
                } catch (Exception ex) {
                    Logger.getLogger(SimulationExperiments.class.getName()).log(Level.SEVERE, null, ex);
                }

            }

            else if (normalize) {

                RowNormalizer nc = new RowNormalizer();
                split[0] = nc.transform(split[0]);
                split[1] = nc.transform(split[1]);
            }

            double acc=singleSampleExperiment(split[0],split[1],c,fold,predictions);
//            System.out.println("simulator ="+simulator+" Classifier ="+classifier+" Fold "+fold+" Acc ="+acc);
            return acc;
        }
        else
            System.out.println(predictions+"/testFold"+fold+".csv already exists");
 //       of.writeString("\n");
        return -1;
    }        
    public static void pairwiseTests(){
    
    }
    public static void combineTestResults(String classifier, String simulator,String resultsPath){
        int folds=200;
        File f=new File(resultsPath+"/"+simulator);
        if(!f.exists() || !f.isDirectory()){
            f.mkdir();
        }
        else{
            boolean results=false;
            for(int i=0;i<folds && !results;i++){
    //Check fold exists            
                f= new File(resultsPath+"/"+simulator+"/"+classifier+"/testFold"+i+".csv");
                if(f.exists())
                    results=true;
            }

            if(results){
                OutFile of=new OutFile(resultsPath+"/"+simulator+"/"+classifier+".csv");
                for(int i=0;i<folds;i++){
        //Check fold exists            
                    f= new File(resultsPath+"/"+simulator+"/"+classifier+"/testFold"+i+".csv");
                    if(f.exists() && f.length()>0){
                        InFile inf=new InFile(resultsPath+"/"+simulator+"/"+classifier+"/testFold"+i+".csv");
                        inf.readLine();
                        inf.readLine();
                        of.writeLine(i+","+inf.readDouble());
                    }
                }
                of.closeFile();
            }
        }
    }
    public static double singleSampleExperiment(Instances train, Instances test, Classifier c, int sample,String preds){
        double acc=0;
        OutFile p=new OutFile(preds+"/testFold"+sample+".csv");

// hack here to save internal CV for further ensembling   
//        if(c instanceof TrainAccuracyEstimate)
//            ((TrainAccuracyEstimate)c).writeCVTrainToFile(preds+"/trainFold"+sample+".csv");
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

    public static void collateAllResults(){
        String resultsPath="C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\BasicExperiments\\";
        for(String s:allClassifiers){
            for(String a:allSimulators){
//            String a="WholeSeriesElastic";
                combineTestResults(s,a,resultsPath);
            }
        }
        int folds=200;
        for(String a:allSimulators){
            if(new File(resultsPath+a).exists()){
                System.out.println(" Simulation = "+a);
                OutFile of=new OutFile(resultsPath+a+"CombinedResults.csv");
                InFile[] ins=new InFile[allClassifiers.length];
                int count=0;
                of.writeString(",");
                for(String s:allClassifiers){
                    File f=new File(resultsPath+a+"\\"+s+".csv");
                    if(f.exists()){
                        InFile inf=new InFile(resultsPath+a+"\\"+s+".csv");
                        int lines=inf.countLines();
                        if(lines>=folds){
                            System.out.println(" Doing "+a+" and "+s);
                            of.writeString(s+",");
                            ins[count++]=new InFile(resultsPath+a+"\\"+s+".csv");
                        }
                    }
                }
                of.writeString("\n");
                for(int i=0;i<folds;i++){
                    of.writeString("Rep"+i+",");
                    for(int j=0;j<count;j++){
                        ins[j].readInt();
                        double acc=ins[j].readDouble();
                        of.writeString(acc+",");
                    }
                    of.writeString("\n");
                }
            }
        }
   }
/** 
 * FINAL VERSION
 * Stand alone method to exactly reproduce shapelet experiment which 
 we normally 
 */    
    public static void runShapeletSimulatorExperiment(){
        Model.setDefaultSigma(1);
        seriesLength=300;
        casesPerClass=new int[]{50,50};
        String[] classifiers={"RotF","DTW","FastShapelets","ST","BOSS"};
//            "EE","CAWPE","TSF","TSBF","FastShapelets","ST","LearnShapelets","BOP","BOSS","C_RISE","COTE"};
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
                            ((ShapeletTransformClassifier)c).setOneMinuteLimit();
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

/** Method to run the error experiment, default setttings 
 * 
 */    
    public static void runErrorExperiment(String[] args){
        String simulator=args[0];
        String classifier=args[1];
        int e=Integer.parseInt(args[2])-1;
        int fold=Integer.parseInt(args[3])-1;
        String resultsPath=args[4];
//Set up the train and test files
        File f=new File(resultsPath+"Error");
        if(!f.exists())
            f.mkdir();
        f=new File(resultsPath+"Error/"+simulator);
        if(!f.exists())
            f.mkdir();
        String predictions= resultsPath+"Error/"+simulator+"/"+classifier;
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
        //E encodes the error and the job number. So 
        double error=((double)e/10.0);
        f=new File(predictions+"/testAcc"+e+"_"+fold+".csv");
        if(!f.exists() || f.length()==0){
            setStandardGlobalParameters(simulator);            
            Model.setDefaultSigma(error);
            Instances data=simulateData(simulator,50*(e+1)*fold);
            Classifier c=setClassifier(classifier);
            Instances[] split=InstanceTools.resampleInstances(data, fold,trainProp);
            double a=ClassifierTools.singleTrainTestSplitAccuracy(c,split[0],split[1]);
            OutFile out=new OutFile(predictions+"/testAcc"+e+"_"+fold+".csv");
            out.writeLine(a+"");
        }
    }

    
    
/** Method to run the error experiment, default setttings 
 * 
 */    
    public static void runLengthExperiment(String[] args){
        String simulator=args[0];
        String classifier=args[1];
//Series length factor
        int l=Integer.parseInt(args[2]);
        String resultsPath=args[3];

        seriesLength=10+(1+l)*50;   //l from 1 to 50
//Set up the train and test files
        File f=new File(resultsPath+simulator+"Length");
        if(!f.exists())
            f.mkdir();
        String predictions= resultsPath+simulator+"Length/"+classifier;
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();

//Check whether fold already exists, if so, dont do it, just quit
        f=new File(predictions+"/testAcc"+l+".csv");
        if(!f.exists() || f.length()==0){
//Do the experiment: just measure the single fold accuracy
            OutFile out=new OutFile(predictions+"/testAcc"+l+".csv");
            double acc=0;
            double var=0;
            for(int fold=0;fold<100;fold++){
                Instances data=simulateData(simulator,seriesLength);
                Classifier c=setClassifier(classifier);
                Instances[] split=InstanceTools.resampleInstances(data, fold,0.5);
                double a=ClassifierTools.singleTrainTestSplitAccuracy(c,split[0],split[1]);
                acc+=a;
                var+=a*a;
            }
            out.writeLine(acc/100+","+var);
        }
    }
    

    public static void trainSetSizeExperiment(String[] args){
        String simulator=args[0];
        String classifier=args[1];
//Series length factor
        int l=Integer.parseInt(args[2]);
        String resultsPath=args[3];
        
        trainProp=(double)(l/10.0);   //l from 1 to 9
//Set up the train and test files
        File f=new File(resultsPath+simulator+"Length");
        if(!f.exists())
            f.mkdir();
        String predictions= resultsPath+simulator+"Length/"+classifier;
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();

//Check whether fold already exists, if so, dont do it, just quit
        f=new File(predictions+"/testAcc"+l+".csv");
        if(!f.exists() || f.length()==0){
//Do the experiment: just measure the single fold accuracy
            OutFile out=new OutFile(predictions+"/testAcc"+l+".csv");
            double acc=0;
            double var=0;
            for(int fold=0;fold<100;fold++){
                Instances data=simulateData(simulator,50*(l+1)*fold);
                Classifier c=setClassifier(classifier);
                Instances[] split=InstanceTools.resampleInstances(data, fold,0.5);
                double a=ClassifierTools.singleTrainTestSplitAccuracy(c,split[0],split[1]);
                acc+=a;
                var+=a*a;
            }
            out.writeLine(acc/100+","+var);
        }
    }
    
  //<editor-fold defaultstate="collapsed" desc="One off data processing methods">       
    public static void collateErrorResults(){
        String path="C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\Error\\";
        double[][] means=new double[allClassifiers.length][21];
        for(String a:allSimulators){
            OutFile out=new OutFile(path+"CollatedError"+a+"Results.csv");
            for(String s:allClassifiers)
                out.writeString(","+s);
            out.writeString("\n");
            int count=0;
            for(String s:allClassifiers){
                for(int i=0;i<=20;i++){
                    int x=0;
                    for(int j=0;j<100;j++){
                        File f= new File(path+a+"\\"+s+"\\"+"testAcc"+i+"_"+j+".csv");
                        if(f.exists() && f.length()>0){
                            InFile inf=new InFile(path+a+"\\"+s+"\\"+"testAcc"+i+"_"+j+".csv");
                            double aa=inf.readDouble();
                            means[count][i]+=aa;
                            x++;
                        }
                    }
                    if(x>0)
                        means[count][i]/=x;
                }
                count++;
            }
            for(int j=0;j<means[0].length;j++){
                out.writeString(100*(j+1)+",");
                for(int i=0;i<means.length;i++)
                    out.writeString(means[i][j]+",");
                out.writeString("\n");
            }
        }
    }
      
    public static void collateLengthResults(){
        String path="C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\ShapeletLength\\";
        OutFile out=new OutFile(path+"CollatedLengthResults.csv");
        out.writeString("Error");
        for(String s:allClassifiers)
            out.writeString(","+s);
        out.writeString("\n");
        for(int i=0;i<10;i++){
            out.writeString((i*50+10)+"");
            for(String s:allClassifiers){
                File f= new File(path+s+"\\"+"testAcc"+i+".csv");
                if(f.exists() && f.length()>0){
                    InFile inf=new InFile(path+s+"\\"+"testAcc"+i+".csv");
                    double a=inf.readDouble();
                    out.writeString(","+a);
                }
                else
                    out.writeString(",");
            }
            out.writeString("\n");
        }       
    }
      

    public static void createBaseExperimentScripts(boolean grace){

//Generates cluster scripts for all combos of classifier and simulator     
       String path="C:\\Users\\ajb\\Dropbox\\Code\\Cluster Scripts\\SimulatorScripts\\BaseExperiment\\";
       File f=new File(path);
       int folds=200; 
       if(!f.isDirectory())
           f.mkdir();
        for(String a:allSimulators){
            OutFile of2;
            if(grace)
                of2=new OutFile(path+a+"Grace.txt");
            else
                of2=new OutFile(path+a+".txt");
            for(String s:allClassifiers){
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
                of.writeLine("#BSUB -R \"rusage[mem=2000]\"");
                of.writeLine("#BSUB -M 2000");
                of.writeLine(" module add java/jdk/1.8.0_31");
            }
            else{
                of.writeLine("#BSUB -R \"rusage[mem=6000]\"");
                of.writeLine("#BSUB -M 6000");
                of.writeLine("module add java/jdk1.8.0_51");
            }
                of.writeLine("java -jar Simulator.jar "+a+" "+ s+" $LSB_JOBINDEX");                
                if(grace)
                    of2.writeLine("bsub < Scripts/SimulatorExperiments/BaseExperiment/"+s+a+"Grace.bsub");
                else
                    of2.writeLine("bsub < Scripts/SimulatorExperiments/BaseExperiment/"+s+a+".bsub");
            }   
        }
    } 
    
    public static void createErrorScripts(boolean grace,String simulator){

//Generates cluster scripts for all combos of classifier and simulator     
       String path="C:\\Users\\ajb\\Dropbox\\Code\\Cluster Scripts\\SimulatorScripts\\Error\\";
       File f= new File(path+simulator);
       if(!f.isDirectory())
           f.mkdir();
//        for(String simulator:allSimulators)
       String ext;
       if(grace)
           ext="ErrorGrace";
       else
           ext="Error";
        for(String classifier:allClassifiers){
            OutFile of2=new OutFile(path+simulator+classifier+ext+".txt");
            for(int i=1;i<=21;i++){
    //                OutFile of = new OutFile(path+"OC"+classifier+simulator+ext+".bsub");
                OutFile of = new OutFile(path+simulator+"\\"+"\\"+classifier+"_"+ext+"_"+i+".bsub");
                of.writeLine("#!/bin/csh");
                if(grace)
                    of.writeLine("#BSUB -q short");
                 else
                    of.writeLine("#BSUB -q long-eth");
                of.writeLine("#BSUB -J "+classifier+"[1-100]");
                of.writeLine("#BSUB -oo output/"+simulator+".out");
                of.writeLine("#BSUB -eo error/"+simulator+".err");
                of.writeLine("#BSUB -R \"rusage[mem=6000]\"");
                of.writeLine("#BSUB -M 6000");
                if(grace)
                    of.writeLine(" module add java/jdk/1.8.0_31");
                else
                   of.writeLine("module add java/jdk1.8.0_51");
                of.writeLine("java -jar Error.jar "+simulator+" "+ classifier+" "+i+" "+ "$LSB_JOBINDEX");                

                of2.writeLine("bsub < Scripts/SimulatorExperiments/Error/"+simulator+"/"+classifier+"_"+ext+"_"+i+".bsub");
            }   
        }
    } 
    
    public static void collateSingleFoldErrorResults(){
        String classifier="LearnShapelets";
        String path="C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\ShapeletError\\"+classifier+"\\";
        OutFile of=new OutFile(path+classifier+".csv");
        for(int i=0; i<21;i++){
            double mean=0;
            for(int folds=0;folds<100;folds++){
                int index=i*100+folds;
                InFile inf=new InFile(path+"testAcc"+index+"_"+folds+".csv");
                mean+=inf.readDouble();
            }
            mean/=100;
            of.writeLine(i+","+mean);
        }
    }

    public static void collateSomeStuff(){
        String[] classifiers={"RotF","DTW","BOSS","ST"};
        for(String str:classifiers){
            String path="C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\Dictionary\\"+str+"\\";
            OutFile of=new OutFile(path+str+".csv");
            double mean=0;
            for(int folds=0;folds<200;folds++){
                File f=new File(path+"testFold"+folds+".csv");
                if(f.exists() && f.length()>0){
                    InFile inf=new InFile(path+"testFold"+folds+".csv");
                    inf.readLine();
                    inf.readLine();
                    double x=inf.readDouble();
                    of.writeLine(folds+","+x);
                }
            }
/*            OutFile of2=new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\MatrixProfileExperiments\\Dictionary.csv");
            InFile[] all=new InFile[4];
            for(String str:classifiers){
                
            }        */
        }
    }
  //</editor-fold>

    public static void generateAllProblemFiles(){
        for(String sim:allSimulators)
            generateProblemFile(sim);
    }
    public static void generateProblemFile(String sim){
        setStandardGlobalParameters(sim);
        int s=22;
        Model.setGlobalRandomSeed(s);
        Model.setDefaultSigma(0.2);
        casesPerClass=new int[]{50,50};
        try{
            Instances data=simulateData(sim,s);
            Instances[] split=InstanceTools.resampleInstances(data, 0,trainProp);
            OutFile train=new OutFile("c:\\temp\\"+sim+"SimLowNoise.csv");
            train.writeString(split[0].toString());
            Model.setDefaultSigma(1);
            data=simulateData(sim,1);
    //        data=SimulateDictionaryData.generateDictionaryData(seriesLength,casesPerClass);
            split=InstanceTools.resampleInstances(data, 0,trainProp);
            train=new OutFile("c:\\temp\\"+sim+"SimNormalNoise.csv");
            train.writeString(split[0].toString());
        }catch(Exception e){
            System.out.println("should do something really ....");
        }
        
    }
    public static void smoothingTests(){
        String sim="WholeSeries";
        setStandardGlobalParameters(sim);
        seriesLength=1000;
        int s=22;
        Model.setGlobalRandomSeed(s);
        Model.setDefaultSigma(5);
        casesPerClass=new int[]{50,50};
        String[] names={"ED","DTW","RotF","BOSS","TSF"};//,"ST","CAWPE","HIVECOTE"};
        Classifier[] cls=new Classifier[names.length];
        for(int i=0;i<names.length;i++)
            cls[i]=setClassifier(names[i]);
        try{
            Instances data=simulateData(sim,s);
            addSpikes(data);
            Instances[] split=InstanceTools.resampleInstances(data, 0,trainProp);
            DecimalFormat df= new DecimalFormat("##.##");
            for(int i=0;i<names.length;i++){
                double d=ClassifierTools.singleTrainTestSplitAccuracy(cls[i], split[0], split[1]);
                System.out.println(names[i]+" acc = "+df.format(d));
            }
        }catch(Exception e){
            System.out.println("should do something really ....");
        }
  }
    public static void addSpikes(Instances t){
        double peak=100;
        int numSpikes=10;
        for(int i=0;i<numSpikes;i++){
            for(Instance ins:t){
                    int position=Model.rand.nextInt(t.numAttributes()-1);
//                if(Model.rand.nextDouble()<0.5){
                    
                    if(Model.rand.nextDouble()<0.5)
                        ins.setValue(position, peak);
                    else
                        ins.setValue(position, -peak);

//                    }
            }
        }
    }
    public static void main(String[] args) throws Exception{
        collateSimulatorResults();
   //     dictionarySimulatorChangingSeriesLength();
  //    dictionarySimulatorChangingTrainSize();
        System.exit(0);

        smoothingTests();
        String resultsPath="C:/Temp/";

        System.exit(0);
        if(args.length>0){
            if(args.length==3){//Base experiment
                double b=runSimulationExperiment(args,true);
                System.out.println(args[0]+","+args[1]+","+","+args[2]+" Acc ="+b);
            }else if(args.length==4){//Error experiment)
                runErrorExperiment(args);
                
            }
//              runLengthExperiment(paras);
        }
        else{
//            DatasetLists.resultsPath="C:\\Users\\ajb\\Dropbox\\Results\\MatrixProfileExperiments\\";
            local=true;
            String[] algos={"ED"};//,,"MP_RotF","MP_DTW"};
            double[] meanAcc=new double[algos.length];
                
            for(int i=1;i<=10;i++){
                for(int j=0;j<algos.length;j++){
                setStandardGlobalParameters("WholeSeries");
                Model.setDefaultSigma(20);
                    String[] para={"WholeSeries",algos[j],i+""};
                    double b=runSimulationExperiment(para,false);
                    meanAcc[j]+=b;
                    System.out.println(para[0]+","+para[1]+","+","+para[2]+" Acc ="+b);
                    
                }
            } 
            DecimalFormat df=new DecimalFormat("##.####");
            for(int j=0;j<algos.length;j++)
                System.out.println(algos[j]+" mean acc ="+df.format(meanAcc[j]/10));
        }
    }

    public static void dictionarySimulatorChangingTrainSize() throws Exception {
        Model.setDefaultSigma(1);
        boolean overwrite=false;
        int seriesLength = 1000;
        int experiments=2;
        String writePath="Z:/Results Working Area/DictionaryBased/SimulationExperimentsMemMonitor2/";
        for(int trainSize=500;trainSize<=10000;trainSize+=500) {
            File path = new File(writePath + "DictionaryTrainSize" + trainSize);
            path.mkdirs();
            if(!overwrite) {
                File f1 = new File(writePath + "DictionaryTrainSize" + trainSize + "/testAcc" + trainSize + ".csv");
                File f2 = new File(writePath + "DictionaryTrainSize" + trainSize + "/trainTime" + trainSize + ".csv");
                File f3 = new File(writePath + "DictionaryTrainSize" + trainSize + "/testTime" + trainSize + ".csv");
                File f4 = new File(writePath + "DictionaryTrainSize" + trainSize + "/mem" + trainSize + ".csv");
                if(f1.exists() && f2.exists() && f3.exists() && f4.exists()){
                    System.out.println("SKIPPING train size = "+trainSize+" as all already present");
                    continue;
                }


            }
            OutFile accFile = new OutFile(writePath + "DictionaryTrainSize" + trainSize  + "/testAcc" + trainSize + ".csv");
            OutFile trainTimeFile = new OutFile(writePath + "DictionaryTrainSize" + trainSize +"/trainTime" + trainSize + ".csv");
            OutFile testTimeFile = new OutFile(writePath + "DictionaryTrainSize" + trainSize  + "/testTime" + trainSize + ".csv");
            OutFile memFile = new OutFile(writePath + "DictionaryTrainSize" + trainSize  + "/mem" + trainSize + ".csv");
            System.out.println(" Generating simulated data for n ="+trainSize+" Series Length ="+seriesLength+" ....");
            int[] casesPerClass = new int[2];
            casesPerClass[0] = casesPerClass[1] = trainSize;
            int[] shapesPerClass = new int[]{5, 20};
            long t1, t2;
            String[] classifierNames = {"cBOSS", "BOSS","WEASEL","S-BOSS"};
            double[] acc = new double[classifierNames.length];
            long[] trainTime = new long[classifierNames.length];
            long[] testTime = new long[classifierNames.length];
            long[] finalMem = new long[classifierNames.length];
            long[] maxMem = new long[classifierNames.length];
            for (int i = 0; i < experiments; i++) {
                Instances data = SimulateDictionaryData.generateDictionaryData(500, casesPerClass, shapesPerClass);
                Instances[] split = InstanceTools.resampleInstances(data, i, 0.5);
                System.out.println("Series Length =" + seriesLength + " Experiment Index: " + i + " Train size =" + split[0].numInstances() + " test size =" + split[1].numInstances());
                for (int j = 0; j < classifierNames.length; j++) {
                    System.gc();
                    MemoryMonitor monitor=new MemoryMonitor();
                    monitor.installMonitor();
                    long memoryBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
                    Classifier c = ClassifierLists.setClassifierClassic(classifierNames[j], i);
                    t1 = System.nanoTime();
                    c.buildClassifier(split[0]);
                    trainTime[j] = System.nanoTime() - t1;
                    t1 = System.nanoTime();
                    acc[j] = ClassifierTools.accuracy(split[1], c);
                    testTime[j] = System.nanoTime() - t1;
                    System.gc();
                    finalMem[j] = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory() - memoryBefore;
                    maxMem[j]=monitor.getMaxMemoryUsed();
                    System.out.println("\t" + classifierNames[j] + " ACC = " + acc[j] + " Train Time =" + trainTime[j] +
                            " Test Time = " + testTime[j] + " Final Memory = " + finalMem[j]/1000000+" Max Memory ="+maxMem[j]/1000000);
                }
                accFile.writeString(i + "");
                for (int j = 0; j < classifierNames.length; j++)
                    accFile.writeString("," + acc[j]);
                accFile.writeString("\n");
                trainTimeFile.writeString(i + "");
                for (int j = 0; j < classifierNames.length; j++)
                    trainTimeFile.writeString("," + trainTime[j]);
                trainTimeFile.writeString("\n");
                testTimeFile.writeString(i + "");
                for (int j = 0; j < classifierNames.length; j++)
                    testTimeFile.writeString("," + testTime[j]);
                testTimeFile.writeString("\n");
                memFile.writeString(i + "");
                for (int j = 0; j < classifierNames.length; j++) {
                    memFile.writeString("," + finalMem[j]);
                }
                memFile.writeString(",");
                for (int j = 0; j < classifierNames.length; j++) {
                    memFile.writeString("," + maxMem[j]);
                }
                memFile.writeString("\n");
            }
        }

    }


    public static void dictionarySimulatorChangingSeriesLength() throws Exception {
        Model.setDefaultSigma(1);
        boolean overwrite=true;
        int experiments=2;
        int numCases=2000;
        String writePath="Z:/Results Working Area/DictionaryBased/SimulationExperimentsMemMonitor/";
        for(int seriesLength=5000;seriesLength<=10000;seriesLength+=5000) {
            String dir="Cases1000SeriesLength";
            File path = new File(writePath +dir+ seriesLength);
            path.mkdirs();
            if(!overwrite) {
                File f1 = new File(writePath + dir + seriesLength + "/testAcc" + seriesLength + ".csv");
                File f2 = new File(writePath + dir + seriesLength + "/trainTime" + seriesLength + ".csv");
                File f3 = new File(writePath + dir + seriesLength + "/testTime" + seriesLength + ".csv");
                File f4 = new File(writePath + dir + seriesLength + "/mem" + seriesLength + ".csv");
                if(f1.exists() && f2.exists() && f3.exists() && f4.exists()){
                    System.out.println("SKIPPING series length = "+seriesLength+" as all already present");
                    continue;
                }

            }
            OutFile accFile = new OutFile(writePath + "DictionarySeriesLength" + seriesLength  + "/testAcc" + seriesLength + ".csv");
            OutFile trainTimeFile = new OutFile(writePath + "DictionarySeriesLength" + seriesLength +"/trainTime" + seriesLength + ".csv");
            OutFile testTimeFile = new OutFile(writePath + "DictionarySeriesLength" + seriesLength  + "/testTime" + seriesLength + ".csv");
            OutFile memFile = new OutFile(writePath + "DictionarySeriesLength" + seriesLength  + "/mem" + seriesLength + ".csv");
            System.out.println(" Generating simulated data ....");
            int[] casesPerClass = new int[2];

            casesPerClass[0] = casesPerClass[1] = numCases/2;
            int[] shapesPerClass = new int[]{5, 20};
            long t1, t2;
            String[] classifierNames = {"cBOSS","S-BOSS","WEASEL","BOSS"};
            double[] acc = new double[classifierNames.length];
            long[] trainTime = new long[classifierNames.length];
            long[] testTime = new long[classifierNames.length];
            long[] finalMem = new long[classifierNames.length];
            long[] maxMem = new long[classifierNames.length];
            for (int i = 0; i < experiments; i++) {
                Instances data = SimulateDictionaryData.generateDictionaryData(seriesLength, casesPerClass, shapesPerClass);
                Instances[] split = InstanceTools.resampleInstances(data, i, 0.2);
                System.out.println(" series length =" + seriesLength + " Experiment Index" + i + " Train size =" + split[0].numInstances() + " test size =" + split[1].numInstances());
                for (int j = 0; j < classifierNames.length; j++) {
                    System.gc();
                    MemoryMonitor monitor=new MemoryMonitor();
                    monitor.installMonitor();
                    long memoryBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
                    Classifier c = ClassifierLists.setClassifierClassic(classifierNames[j], i);
                    t1 = System.nanoTime();
                    c.buildClassifier(split[0]);
                    trainTime[j] = System.nanoTime() - t1;
                    t1 = System.nanoTime();
                    acc[j] = ClassifierTools.accuracy(split[1], c);
                    testTime[j] = System.nanoTime() - t1;
                    System.gc();
                    finalMem[j] = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory() - memoryBefore;
                    maxMem[j]=monitor.getMaxMemoryUsed();
                    System.out.println("\t" + classifierNames[j] + " ACC = " + acc[j] + " Train Time =" + trainTime[j] +
                            " Test Time = " + testTime[j] + " Final Memory = " + finalMem[j]/1000000+" Max Memory ="+maxMem[j]/1000000);
                }
                accFile.writeString(i + "");
                for (int j = 0; j < classifierNames.length; j++)
                    accFile.writeString("," + acc[j]);
                accFile.writeString("\n");
                trainTimeFile.writeString(i + "");
                for (int j = 0; j < classifierNames.length; j++)
                    trainTimeFile.writeString("," + trainTime[j]);
                trainTimeFile.writeString("\n");
                testTimeFile.writeString(i + "");
                for (int j = 0; j < classifierNames.length; j++)
                    testTimeFile.writeString("," + testTime[j]);
                testTimeFile.writeString("\n");
                memFile.writeString(i + "");
                for (int j = 0; j < classifierNames.length; j++) {
                    memFile.writeString("," + finalMem[j]);
                }
                memFile.writeString(",");
                for (int j = 0; j < classifierNames.length; j++) {
                    memFile.writeString("," + maxMem[j]);
                }
                memFile.writeString("\n");
            }
        }

    }


    public static void collateSimulatorResults(){
        String type="Dictionary";
        String path="Z:\\Results Working Area\\"+type+"Based\\\\SimulationExperimentsMemMonitor\\";
        File f= new File(path+type+"Summary");
        f.mkdirs();
        String[] files={"testAcc","testTime","trainTime","mem"};
        int numClassifiers=4;
        OutFile[] out=new OutFile[files.length];
        OutFile[] outDiffs=new OutFile[files.length];
        for(int i=0;i<files.length;i++){
            out[i]=new OutFile(path+type+"Summary\\"+files[i]+"Mean.csv");
            out[i].writeLine("Means,BOSS,cBOSS,S-BOSS,WEASEL,StDevs,BOSS,cBOSS,S-BOSS,WEASEL");
            outDiffs[i]=new OutFile(path+type+"Summary\\"+files[i]+"MeanDiffs.csv");
            outDiffs[i].writeLine("MeanDiffsToBOSS,cBOSS,S-BOSS,WEASEL,StDevs,cBOSS,S-BOSS,WEASEL");
        }


        for(int i=0;i<files.length;i++){
            String s=files[i];
            ArrayList<double[]> medians=new ArrayList<>();
            for(int trainSize=50;trainSize<=1000;trainSize+=50) {
                File test;
                int lines = 0;
                String fPath=path + type + "TrainSize" + trainSize + "\\" + s + trainSize + ".csv";
                f = new File(fPath);
                if (!f.exists()) {
                    System.out.println("File " + s + trainSize + " does not exist on" + fPath+"  skipping " + trainSize);
                    continue;

                }

                //How many have we got?
                InFile inf = new InFile(fPath);
                int l = inf.countLines();
                System.out.println(" File = "+fPath);
                System.out.println(trainSize + " has " + l + " lines");
                inf = new InFile(fPath);
                double[][] vals = new double[l][numClassifiers];
                double[][] diffs = new double[l][numClassifiers-1];
                if(files[i].equals("mem"))
                {
                    for (int j = 0; j < l; j++) {
                        String[] line = inf.readLine().split(",");
                        vals[j][0] = Double.parseDouble(line[6]);
                        for (int k = 1; k <numClassifiers; k++) {
                            vals[j][k] = Double.parseDouble(line[k + 6]);
                            diffs[j][k - 1] = vals[j][k] - vals[j][0];
                        }
                    }
                }
                else{
                    for (int j = 0; j < l; j++) {
                        String[] line = inf.readLine().split(",");
                        vals[j][0] = Double.parseDouble(line[1]);
                        for (int k = 1; k < numClassifiers; k++) {
                            vals[j][k] = Double.parseDouble(line[k + 1]);
                            diffs[j][k - 1] = vals[j][k] - vals[j][0];
                        }
                    }
                }
                //Find means
                double[] means = new double[numClassifiers];
                double[] meanDiffs = new double[numClassifiers];
                for (int k = 0; k < numClassifiers; k++) {
                    means[k] = 0;
                    for (int j = 0; j < l; j++) {
                        means[k] += vals[j][k];
                    }
                    means[k] /= l;
                }
                for (int k = 0; k < numClassifiers-1; k++) {
                    meanDiffs[k] = 0;
                    for (int j = 0; j < l; j++) {
                        meanDiffs[k] += diffs[j][k];
                    }
                    meanDiffs[k] /= l;
                }
                double[] confInterval = new double[numClassifiers];
                double[] confIntervalDiffs = new double[numClassifiers];
                for (int k = 0; k < numClassifiers; k++) {
                    confInterval[k] = 0;
                    for (int j = 0; j < l; j++) {
                        confInterval[k] += (vals[j][k]-means[k])*(vals[j][k]-means[k]);
                    }
                    confInterval[k] /= l-1;
                    confInterval[k]=Math.sqrt(confInterval[k]);
                    confInterval[k]/=Math.sqrt(l);
                    confInterval[k]*=1.96;

                }
                for (int k = 0; k < numClassifiers-1; k++) {
                    confIntervalDiffs[k] = 0;
                    for (int j = 0; j < l; j++) {
                        confIntervalDiffs[k] += (diffs[j][k]- meanDiffs[k])*(diffs[j][k]- meanDiffs[k]);
                    }
                    confIntervalDiffs[k] /= (l-1);
                    confIntervalDiffs[k]=Math.sqrt(confIntervalDiffs[k]);
                    confIntervalDiffs[k]/=Math.sqrt(l);
                    confIntervalDiffs[k]*=1.96;
                }


                //Write to file
                if(!s.equals("testTime"))
                    out[i].writeString(trainSize + "");
                else
                    out[i].writeString((int)(0.9*(trainSize/0.1)) + "");
                for (int k = 0; k < numClassifiers; k++) {
                    out[i].writeString("," + means[k]);
                }
                out[i].writeString(",");
                for (int k = 0; k < numClassifiers; k++) {
                    out[i].writeString("," + confInterval[k]);
                }

                out[i].writeString("\n");
                if(!s.equals("testTime"))
                    outDiffs[i].writeString(trainSize + "");
                else
                    outDiffs[i].writeString((int)(0.9*(trainSize/0.1)) + "");
                for (int k = 0; k < numClassifiers-1; k++) {
                    outDiffs[i].writeString("," + meanDiffs[k]);
                }
                outDiffs[i].writeString(",");
                for (int k = 0; k < numClassifiers-1; k++) {
                    outDiffs[i].writeString("," + confIntervalDiffs[k]);
                }
                outDiffs[i].writeString("\n");
            }

            for(int seriesLength=500;seriesLength<=10000;seriesLength+=500) {
                File test;
                int lines = 0;
                String fPath=path + type + "SeriesLength" + seriesLength + "\\" + s + seriesLength + ".csv";
                f = new File(fPath);
                if (!f.exists()) {
                    System.out.println("File " + s + seriesLength + " does not exist on" + fPath+"  skipping " + seriesLength);
                    continue;
                }
                //How many have we got?
                InFile inf = new InFile(fPath);
                int l = inf.countLines();
                System.out.println(seriesLength + " has " + l + " lines");
                inf = new InFile(fPath);
                double[][] vals = new double[l][numClassifiers];
                double[][] diffs = new double[l][numClassifiers-1];
                if(files[i].equals("mem"))
                {
                    for (int j = 0; j < l; j++) {
                        String[] line = inf.readLine().split(",");
                        vals[j][0] = Double.parseDouble(line[6]);
                        for (int k = 1; k <numClassifiers; k++) {
                            vals[j][k] = Double.parseDouble(line[k + 6]);
                            diffs[j][k - 1] = vals[j][k] - vals[j][0];
                        }
                    }
                }
                else{
                    for (int j = 0; j < l; j++) {
                        String[] line = inf.readLine().split(",");
                        vals[j][0] = Double.parseDouble(line[1]);
                        for (int k = 1; k < numClassifiers; k++) {
                            vals[j][k] = Double.parseDouble(line[k + 1]);
                            diffs[j][k - 1] = vals[j][k] - vals[j][0];
                        }
                    }
                }
                //Find means
                double[] means = new double[numClassifiers];
                double[] meanDiffs = new double[numClassifiers];
                for (int k = 0; k < numClassifiers; k++) {
                    means[k] = 0;
                    for (int j = 0; j < l; j++) {
                        means[k] += vals[j][k];
                    }
                    means[k] /= l;
                }
                for (int k = 0; k < numClassifiers-1; k++) {
                    meanDiffs[k] = 0;
                    for (int j = 0; j < l; j++) {
                        meanDiffs[k] += diffs[j][k];
                    }
                    meanDiffs[k] /= l;
                }
                double[] confInterval = new double[numClassifiers];
                double[] confIntervalDiffs = new double[numClassifiers];
                for (int k = 0; k < numClassifiers; k++) {
                    confInterval[k] = 0;
                    for (int j = 0; j < l; j++) {
                        confInterval[k] += (vals[j][k]-means[k])*(vals[j][k]-means[k]);
                    }
                    confInterval[k] /= l-1;
                    confInterval[k]=Math.sqrt(confInterval[k]);
                    confInterval[k]/=Math.sqrt(l);
                    confInterval[k]*=1.96;

                }
                for (int k = 0; k < numClassifiers-1; k++) {
                    confIntervalDiffs[k] = 0;
                    for (int j = 0; j < l; j++) {
                        confIntervalDiffs[k] += (diffs[j][k]- meanDiffs[k])*(diffs[j][k]- meanDiffs[k]);
                    }
                    confIntervalDiffs[k] /= (l-1);
                    confIntervalDiffs[k]=Math.sqrt(confIntervalDiffs[k]);
                    confIntervalDiffs[k]/=Math.sqrt(l);
                    confIntervalDiffs[k]*=1.96;
                }


                //Write to file
                out[i].writeString(seriesLength + "");
                for (int k = 0; k < numClassifiers; k++) {
                    out[i].writeString("," + means[k]);
                }
                out[i].writeString(",");
                for (int k = 0; k < numClassifiers; k++) {
                    out[i].writeString("," + confInterval[k]);
                }

                out[i].writeString("\n");
                outDiffs[i].writeString(seriesLength + "");
                for (int k = 0; k < numClassifiers-1; k++) {
                    outDiffs[i].writeString("," + meanDiffs[k]);
                }
                outDiffs[i].writeString(",");
                for (int k = 0; k < numClassifiers-1; k++) {
                    outDiffs[i].writeString("," + confIntervalDiffs[k]);
                }
                outDiffs[i].writeString("\n");
            }


        }
    }






    public static void dictionarySimulatorThreadExperiment() throws Exception {
        Model.setDefaultSigma(1);
        boolean overwrite=false;
        int experiments=1;
        for(int seriesLength=300;seriesLength<=300;seriesLength+=300) {
            int[] casesPerClass = new int[2];
            casesPerClass[0] = casesPerClass[1] = 100;
            int[] shapesPerClass = new int[]{5, 20};
            double[] acc = new double[4];
            long[] trainTime = new long[4];
            long[] testTime = new long[4];
            long[] mem = new long[4];
            long t1, t2;
            String[] classifierNames = {"BOSS"};//, "cBOSS", "SpatialBOSS", "WEASEL"};

            MemoryMXBean mx= ManagementFactory.getMemoryMXBean();
            Notification notif;
/*            GarbageCollectorMXBean gc=mx.
            // receive the notification emitted by a GarbageCollectorMXBean and set to notif
            synchronized (mx){
                mx.wait();
            }
            notif=mx.get
            String notifType = "TESTY"; //notif.getType();
            if (notifType.equals(GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION)) {
                // retrieve the garbage collection notification information
                CompositeData cd = (CompositeData) notif.getUserData();
                GarbageCollectionNotificationInfo info = GarbageCollectionNotificationInfo.from(cd);


            }
*/
            for (int i = 0; i < experiments; i++) {
                Instances data = SimulateDictionaryData.generateDictionaryData(seriesLength, casesPerClass, shapesPerClass);
                Instances[] split = InstanceTools.resampleInstances(data, i, 0.2);
                System.out.println(" Testing thread model: series length =" + seriesLength + " Experiment Index" + i + " Train size =" + split[0].numInstances() + " test size =" + split[1].numInstances());
                for (int j = 0; j < classifierNames.length; j++) {
                    System.gc();
                    long memoryBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
                    Classifier c = ClassifierLists.setClassifierClassic(classifierNames[j], i);
                    t1 = System.nanoTime();
                    c.buildClassifier(split[0]);
                    trainTime[j] = System.nanoTime() - t1;
                    t1 = System.nanoTime();
                    acc[j] = ClassifierTools.accuracy(split[1], c);
                    testTime[j] = System.nanoTime() - t1;
                    System.gc();
                    mem[j] = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory() - memoryBefore;

                    System.out.println("\t" + classifierNames[j] + " ACC = " + acc[j] + " Train Time =" + trainTime[j] +
                            " Test Time = " + testTime[j] + " Memory = " + mem[j]);
                }
            }
        }

    }


    public static class ThreadExperiment implements Runnable{
        Classifier c;
        Instances train;
       public ThreadExperiment(Classifier c, Instances train){
           this.c=c;
           this.train=train;
       }

        @Override
        public void run() {
           try {
               c.buildClassifier(train);
           }catch(Exception e){
               System.out.println("Classifier threw exception in Thread Experiment");
           }
        }
    }

}
