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
import weka.core.Instance;
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
public class SimulationExperiments {
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
    static String[] allSimulators={"WholeSeriesElastic","Interval","Shapelet","Dictionary","ARMA"};
    
    
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
            case "WholeSeries":
                seriesLength=100;
                trainProp=0.1;
                casesPerClass=new int[]{100,100};
                Model.setDefaultSigma(1);
                ElasticModel.setBaseAndAmp(-2, 4);
                ElasticModel.setWarpPercent(0.4);
 //               SimulateWholeSeriesElastic.
                break;
            case "MatrixProfile":
                seriesLength=150;
                trainProp=0.1;
                casesPerClass=new int[]{50,50};
                Model.setDefaultSigma(1);
                break;
        default:
                throw new RuntimeException(" UNKNOWN SIMULATOR ");
            
        }       
    }
    
    
    public static Instances simulateData(String str,int seed) throws RuntimeException{
        Instances data;
//        for(int:)
        Model.setGlobalRandomSeed(seed);
        switch(str){
            case "ARMA": case "AR": case "SPECTRAL":
                
                  data=SimulateSpectralData.generateSpectralEmbeddedData(seriesLength, casesPerClass);
//                 data=SimulateSpectralData.generateARDataSet(seriesLength, casesPerClass, true);
                break;
            case "Shapelet": 
                data=SimulateShapeletData.generateShapeletData(seriesLength,casesPerClass);
                break;
            case "Dictionary":
                data=SimulateDictionaryData.generateDictionaryData(seriesLength,casesPerClass);
               break; 
            case "Interval":    
                data=SimulateIntervalData.generateIntervalData(seriesLength, casesPerClass);
                break;        
                        
            case "WholeSeries":
                data=SimulateWholeSeriesData.generateWholeSeriesdData(seriesLength,casesPerClass);
                break;
           case "WholeSeriesElastic":
                data=SimulateElasticData.generateElasticData(seriesLength,casesPerClass);
                break;
           case "MatrixProfile":
                data=SimulateMatrixProfileData.generateMatrixProfileData(seriesLength,casesPerClass);
                break;

           
           default:
                throw new RuntimeException(" UNKNOWN SIMULATOR "+str);
            
        }
        return data;
    }
    

//arg[0]: simulator
//arg[1]: classifier
//arg[2]: fold number    
    public static double runSimulationExperiment(String[] args,boolean useStandard) throws Exception{
        String simulator=args[0];
        if(useStandard)
            setStandardGlobalParameters(simulator);
        String classifier=args[1];
        Classifier c=setClassifier(classifier);
        int fold=Integer.parseInt(args[2])-1;


//Set up the train and test files
        File f=new File(DataSets.resultsPath+simulator);
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+simulator+"/"+classifier;
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
//Check whether fold already exists, if so, dont do it, just quit
        f=new File(predictions+"/testFold"+fold+".csv");
        if(!f.exists() || f.length()==0){
//Do the experiment: find train preds through cross validation
//Then generate all test predictions            
            Instances data=simulateData(args[0],fold);
            Instances[] split=InstanceTools.resampleInstances(data, fold,trainProp);
            System.out.println(" Train size ="+split[0].numInstances()+" test size ="+split[1].numInstances());
    //Check if it is MP or not
            if(classifier.contains("MP_")){
                try {
                    System.out.println("MAtrix profile run ....");
                    MatrixProfile mp=new MatrixProfile(29);
                    split[0]=mp.process(split[0]);
                    split[1]=mp.process(split[1]);
                } catch (Exception ex) {
                    Logger.getLogger(SimulationExperiments.class.getName()).log(Level.SEVERE, null, ex);
                }

            }
            
            else if(normalize){
                
                NormalizeCase nc= new NormalizeCase();
                split[0]=nc.process(split[0]);
                split[0]=nc.process(split[1]);
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
    public static void combineTestResults(String classifier, String simulator){
        int folds=200;
        File f=new File(DataSets.resultsPath+"/"+simulator);
        if(!f.exists() || !f.isDirectory()){
            f.mkdir();
        }
        else{
            boolean results=false;
            for(int i=0;i<folds && !results;i++){
    //Check fold exists            
                f= new File(DataSets.resultsPath+"/"+simulator+"/"+classifier+"/testFold"+i+".csv");
                if(f.exists())
                    results=true;
            }

            if(results){
                OutFile of=new OutFile(DataSets.resultsPath+"/"+simulator+"/"+classifier+".csv");
                for(int i=0;i<folds;i++){
        //Check fold exists            
                    f= new File(DataSets.resultsPath+"/"+simulator+"/"+classifier+"/testFold"+i+".csv");
                    if(f.exists() && f.length()>0){
                        InFile inf=new InFile(DataSets.resultsPath+"/"+simulator+"/"+classifier+"/testFold"+i+".csv");
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

    public static void collateAllResults(){
        DataSets.resultsPath="C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\BasicExperiments\\";
        for(String s:allClassifiers){
            for(String a:allSimulators){
//            String a="WholeSeriesElastic";
                combineTestResults(s,a);
            }
        }
        int folds=200;
        for(String a:allSimulators){
            if(new File(DataSets.resultsPath+a).exists()){
                System.out.println(" Simulation = "+a);
                OutFile of=new OutFile(DataSets.resultsPath+a+"CombinedResults.csv");
                InFile[] ins=new InFile[allClassifiers.length];
                int count=0;
                of.writeString(",");
                for(String s:allClassifiers){
                    File f=new File(DataSets.resultsPath+a+"\\"+s+".csv");
                    if(f.exists()){
                        InFile inf=new InFile(DataSets.resultsPath+a+"\\"+s+".csv");
                        int lines=inf.countLines();
                        if(lines>=folds){
                            System.out.println(" Doing "+a+" and "+s);
                            of.writeString(s+",");
                            ins[count++]=new InFile(DataSets.resultsPath+a+"\\"+s+".csv");
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
                            ((ShapeletTransformClassifier)c).setOneMinuteLimit();
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

/** Method to run the error experiment, default setttings 
 * 
 */    
    public static void runErrorExperiment(String[] args){
        String simulator=args[0];
        String classifier=args[1];
        int e=Integer.parseInt(args[2])-1;
        int fold=Integer.parseInt(args[3])-1;
//Set up the train and test files
        File f=new File(DataSets.resultsPath+"Error");
        if(!f.exists())
            f.mkdir();
        f=new File(DataSets.resultsPath+"Error/"+simulator);
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+"Error/"+simulator+"/"+classifier;
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
        seriesLength=10+(1+l)*50;   //l from 1 to 50
//Set up the train and test files
        File f=new File(DataSets.resultsPath+simulator+"Length");
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+simulator+"Length/"+classifier;
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
        trainProp=(double)(l/10.0);   //l from 1 to 9
//Set up the train and test files
        File f=new File(DataSets.resultsPath+simulator+"Length");
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+simulator+"Length/"+classifier;
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
smoothingTests();
System.exit(0);
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
}
