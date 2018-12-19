/**
 *
 * @author ajb
 *local class to run experiments with the UCI data


*/
package development.old_experiments;

import development.DataSets;
import development.Experiments;
import development.RotationForestLimitedAttributes;
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
import timeseriesweka.classifiers.ensembles.elastic_ensemble.ED1NN;
import weka.classifiers.trees.RandomForest;
import vector_classifiers.TunedRandomForest;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;


public class July2017Experiments{
    public static String[] classifiers={"SVM"};
    public static double propInTrain=0.5;
    public static int folds=30;
    public static Random rand=new Random(0);
    static boolean debug=true;
    static boolean generateTrainFiles=true;
    
 
    public static Classifier setClassifier(String classifier, int fold){
//RandF or RotF
        TunedRandomForest randF;
        TunedRotationForest r;
        int[] numTrees;
        switch(classifier){
            case "logistic":
                Logistic log=new Logistic();
                return log;
            case "TunedSVMRBF":
                TunedSVM svm=new TunedSVM();
                svm.setKernelType(TunedSVM.KernelType.RBF);
                svm.optimiseParas(true);
                svm.optimiseKernel(false);
                svm.setBuildLogisticModels(true);
                svm.setSeed(fold);
                return svm;
           case "RotF":
                r=new TunedRotationForest();
                r.setNumIterations(200);
                r.justBuildTheClassifier();
                return r;
            case "RandRotF1":
                RotationForestLimitedAttributes r3=new RotationForestLimitedAttributes();
                r3.setNumIterations(200);
                r3.setMaxNumAttributes(100);
                r3.justBuildTheClassifier();
                return r3;
            case "HESCA":
                String[] names={"RotFCV","RandFOOB","SVM"};
                Classifier[] c=new Classifier[3];
                c[0]=new IBk();
                c[1]=new IBk();
                c[2]=new IBk();
                CAWPE h = new CAWPE();
                h.setClassifiers(c, names, null);
                h.setDebug(true);
                return h;    
            case "RotFCV":
                r = new TunedRotationForest();
                r.setNumIterations(200);
                r.tuneParameters(false);
                r.estimateAccFromTrain(true);
                return r;
            case "RandFCV":
                randF = new TunedRandomForest();
                randF.tuneParameters(false);
                randF.setNumTrees(500);
                randF.debug(debug);
                randF.setSeed(fold);
                randF.setEstimateAcc(true);
                randF.setCrossValidate(true);
                return randF;
            case "RandFOOB":
                randF = new TunedRandomForest();
                randF.tuneParameters(false);
                randF.setNumTrees(500);
                randF.debug(debug);
                randF.setSeed(fold);
                randF.setEstimateAcc(true);
                randF.setCrossValidate(false);
                return randF; 
            case "TunedRandFCV":
//These are the built in defaults, here for clarity only                
                numTrees=new int[]{10,50,100,200,300,400,500,600,700,800,900};
                randF = new TunedRandomForest();
                randF.tuneParameters(true);
                randF.setSeed(fold);
                randF.setEstimateAcc(true);
                randF.setCrossValidate(true);
                randF.setNumTreesRange(numTrees);
                return randF;
            case "TunedRandFOOB":
                numTrees=new int[]{10,50,100,200,300,400,500,600,700,800,900};
               randF = new TunedRandomForest();
                randF.tuneParameters(true);
                randF.setSeed(fold);
                randF.setEstimateAcc(true);
                randF.setCrossValidate(false);//Use OOB
                randF.setNumTreesRange(numTrees);
                return randF;
            case "TunedRotFCV":
//These are the built in defaults, here for clarity only                
                numTrees=new int[]{10,50,100,200,300,400,500,600,700,800,900};
                r = new TunedRotationForest();
                r.tuneParameters(true);
                r.setSeed(fold);
                r.setNumTreesRange(numTrees);
                return r;
                
            default:
            throw new RuntimeException("Unknown classifier = "+classifier+" in Feb 2017 class");
        }
    }
    public static void singleClassifierAndFoldSingleDataSet(String[] args){
//first gives the problem file      
        String classifier=args[0];
        String problem=args[1];
        int fold=Integer.parseInt(args[2])-1;
   
        Classifier c=July2017Experiments.setClassifier(classifier,fold);
        Instances all=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem);
//        all.randomize(rand);
        
        Instances[] split=InstanceTools.resampleInstances(all, fold, propInTrain);
        File f=new File(DataSets.resultsPath+classifier);
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+classifier+"/Predictions"+"/"+problem;
        f=new File(predictions);
        if(!f.exists())
            f.mkdirs();
//Check whether fold already exists, if so, dont do it, just quit
        f=new File(predictions+"/testFold"+fold+".csv");
        if(!f.exists() || f.length()==0){
      //      of.writeString(problem+","); );
            if(c instanceof TrainAccuracyEstimate)
                ((TrainAccuracyEstimate)c).writeCVTrainToFile(predictions+"/trainFold"+fold+".csv");
            if(c instanceof CAWPE){
                System.out.println("Turning on file read ");
                  ((CAWPE)c).setResultsFileLocationParameters(DataSets.resultsPath, problem, fold);
                  ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
            }
            double acc =singleClassifierAndFoldSingleDataSet(split[0],split[1],c,fold,predictions);
            System.out.println(classifier+","+problem+","+fold+","+acc);
            
 //       of.writeString("\n");
        }
    }
    
    public static double singleClassifierAndFoldSingleDataSet(Instances train, Instances test, Classifier c, int fold,String resultsPath){
        double acc=0;
        int act;
        int pred;
// Save internal info for ensembles
//        if(c instanceof SaveableEnsemble)
//           ((SaveableEnsemble)c).saveResults(resultsPath+"/internalCV_"+fold+".csv",resultsPath+"/internalTestPreds_"+fold+".csv");
        OutFile p=null;
        try{              
            c.buildClassifier(train);
            StringBuilder str = new StringBuilder();
            DecimalFormat df=new DecimalFormat("##.######");
            for(int j=0;j<test.numInstances();j++)
            {
                act=(int)test.instance(j).classValue();

                test.instance(j).setClassMissing();//Just in case ....
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
                str.append(",");
                for(double d:probs){
                    str.append(",");
                    str.append(df.format(d));
                }
                str.append("\n");
            }
            acc/=test.numInstances();
           
            p=new OutFile(resultsPath+"/testFold"+fold+".csv");
            if(p==null) throw new Exception(" file wont open!! "+resultsPath+"/testFold"+fold+".csv");
            p.writeLine(train.relationName()+","+c.getClass().getName()+",test");
            if(c instanceof SaveParameterInfo){
              p.writeLine(((SaveParameterInfo)c).getParameters());
            }else
                p.writeLine("No parameter info");
            p.writeLine(acc+"");
            p.writeLine(str.toString());
        }catch(Exception e)
        {
                e.printStackTrace();
                System.out.println(" Error ="+e+" in method singleClassifierAndFold in class Feb2017");
                System.out.println(" Classifier = "+c.getClass().getName());
                System.out.println(" Results path="+resultsPath);
                System.out.println(" Outfile = "+p);
                System.out.println(" Train Split = "+train.toSummaryString());
                System.out.println(" Test Split = "+test.toSummaryString());
                e.printStackTrace();
                System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
                System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes"+test.numInstances()+" instances");

                System.exit(0);
        }
         return acc;
    }    
    
/** Run a given classifier/problem/fold combination with associated file set up
 @param args: 
 * args[0]: Classifier name. Create classifier with setClassifier
 * args[1]: Problem name
 * args[2]: Fold number. This is assumed to range from 1, hence we subtract 1
 * (this is because of the scripting we use to run the code on the cluster)
 *          the standard archive folds are always fold 0
 * 
 * NOTES: 
 * 1. this assumes you have set DataSets.problemPath to be where ever the 
 * data is, and assumes the data is in its own directory with two files, 
 * args[1]_TRAIN.arff and args[1]_TEST.arff 
 * 2. assumes you have set DataSets.resultsPath to where you want the results to
 * go It will NOT overwrite any existing results (i.e. if a file of non zero 
 * size exists)
 * 3. This method just does the file set up then calls the next method. If you 
 * just want to run the problem, go to the next method
* */
    public static void singleClassifierAndFoldTrainTestSplit(String[] args){
//first gives the problem file      
        String classifier=args[0];
        String problem=args[1];
        int fold=Integer.parseInt(args[2])-1;
   
        Classifier c=setClassifier(classifier,fold);
        Instances train=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TRAIN");
        Instances test=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TEST");
        File f=new File(DataSets.resultsPath+classifier);
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+classifier+"/Predictions";
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
        predictions=predictions+"/"+problem;
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
//Check whether fold already exists, if so, dont do it, just quit
        f=new File(predictions+"/testFold"+fold+".csv");
        if(!f.exists() || f.length()==0){
      //      of.writeString(problem+","); );
            if(c instanceof TrainAccuracyEstimate)
                ((TrainAccuracyEstimate)c).writeCVTrainToFile(predictions+"/trainFold"+fold+".csv");
            double acc =singleClassifierAndFoldTrainTestSplit(train,test,c,fold,predictions);
            System.out.println(classifier+","+problem+","+fold+","+acc);
            
 //       of.writeString("\n");
        }
    }
/**
 * 
 * @param train: the standard train fold Instances from the archive 
 * @param test: the standard test fold Instances from the archive
 * @param c: Classifier to evaluate
 * @param fold: integer to indicate which fold. Set to 0 to just use train/test
 * @param resultsPath: a string indicating where to store the results
 * @return the accuracy of c on fold for problem given in train/test
 * 
 * NOTES:
 * 1.  If the classifier is a SaveableEnsemble, then we save the internal cross 
 * validation accuracy and the internal test predictions
 * 2. The output of the file testFold+fold+.csv is
 * Line 1: ProblemName,ClassifierName, train/test
 * Line 2: parameter information for final classifier, if it is available
 * Line 3: test accuracy
 * then each line is
 * Actual Class, Predicted Class, Class probabilities 
 * 
 * 
 */    
    public static double singleClassifierAndFoldTrainTestSplit(Instances train, Instances test, Classifier c, int fold,String resultsPath){
        Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, fold);
        double acc=0;
        int act;
        int pred;
// Save internal info for ensembles
        if(c instanceof SaveableEnsemble)
           ((SaveableEnsemble)c).saveResults(resultsPath+"/internalCV_"+fold+".csv",resultsPath+"/internalTestPreds_"+fold+".csv");
        try{              
            c.buildClassifier(data[0]);
            if(debug){
                if(c instanceof RandomForest)
                    System.out.println(" Number of features in MAIN="+((RandomForest)c).getNumFeatures());
            }
            StringBuilder str = new StringBuilder();
            DecimalFormat df=new DecimalFormat("##.######");
            for(int j=0;j<data[1].numInstances();j++)
            {
                act=(int)data[1].instance(j).classValue();
                data[1].instance(j).setClassMissing();//Just in case ....
                double[] probs=c.distributionForInstance(data[1].instance(j));
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
            acc/=data[1].numInstances();
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

  public static void main(String[] args) throws Exception{
      sanityCheckBones();
      System.exit(0);
              
        for(String str:args)
            System.out.println(str);
        if(args.length!=6){//Local run
            DataSets.problemPath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\";
            DataSets.resultsPath="c:\\Temp\\";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
            }
            generateTrainFiles=true;
            String[] newArgs={"RandF","ItalyPowerDemand","1"};
            Experiments.singleClassifierAndFoldTrainTestSplit(newArgs);
            System.exit(0);
        }
        else{    
            DataSets.problemPath=args[0];
            DataSets.resultsPath=args[1];
    //Third argument is whether to cross validate or not
            generateTrainFiles=Boolean.parseBoolean(args[2]);
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
            }
            String[] newArgs=new String[args.length-3];
            for(int i=3;i<args.length;i++)
                newArgs[i-3]=args[i];
            Experiments.singleClassifierAndFoldTrainTestSplit(newArgs);
        }    }

    
    public static void runTSCDataSet(String[] args) {
        if(args.length>0){//Cluster run
            DataSets.problemPath=DataSets.clusterPath+"TSCProblems/";
            DataSets.resultsPath=DataSets.clusterPath+"Results/RepoResults/";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
            }
            July2017Experiments.singleClassifierAndFoldTrainTestSplit(args);
        }
        else{
            DataSets.problemPath=DataSets.dropboxPath+"TSC Problems/";
            DataSets.resultsPath=DataSets.dropboxPath+"Results/RepoResults/";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
            }

            String[] paras={"RandFCV","ItalyPowerDemand","1"};
//            paras[0]="RotFCV";
//            paras[2]="1";
            July2017Experiments.singleClassifierAndFoldTrainTestSplit(paras);            
            long t1=System.currentTimeMillis();
            for(int i=2;i<=11;i++){
                paras[2]=i+"";
                July2017Experiments.singleClassifierAndFoldSingleDataSet(paras);            
            }
            long t2=System.currentTimeMillis();
            paras[0]="RandFOOB";
            July2017Experiments.singleClassifierAndFoldSingleDataSet(paras);            
            long t3=System.currentTimeMillis();
            for(int i=2;i<=11;i++){
                paras[2]=i+"";
                July2017Experiments.singleClassifierAndFoldSingleDataSet(paras);            
            }
            long t4=System.currentTimeMillis();
            System.out.println("Standard = "+(t2-t1)+", Enhanced = "+(t4-t3));
            
       }        
    }

    
    public static void sanityCheckBones(){
        String[] files={"MiddlePhalanxOutlineAgeGroup","MiddlePhalanxTW","MiddlePhalanxOutlineCorrect","DistalPhalanxOutlineCorrect","DistalPhalanxOutlineAgeGroup","DistalPhalanxTW"};
        
        for(String str:files){
            Instances train = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+str+"\\"+str+"_TRAIN");
            Instances test = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+str+"\\"+str+"_TEST");
            Classifier c=new ED1NN();
            double a=ClassifierTools.singleTrainTestSplitAccuracy(c, train, test);
            System.out.println(str+" correct ACC = "+a);
            Instances train2 = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+str+"\\"+str+"_TEST");
            Instances test2 = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+str+"\\"+str+"_TRAIN");
            Classifier c2=new ED1NN();
            double a2=ClassifierTools.singleTrainTestSplitAccuracy(c2, train2, test2);
            System.out.println(str+" inverted ACC = "+a2);
        }
        
    }
    public static void runUCIDataSet(String[] args) {
        if(args.length>0){//Cluster run
            DataSets.problemPath=DataSets.clusterPath+"UCIContinuous/";
            DataSets.resultsPath=DataSets.clusterPath+"Results/UCIResults/";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
            }
            July2017Experiments.singleClassifierAndFoldSingleDataSet(args);
        }
        else{
            DataSets.problemPath=DataSets.dropboxPath+"UCI Problems/";
            DataSets.resultsPath=DataSets.dropboxPath+"Results/UCIResults/";
            File f=new File(DataSets.resultsPath);
            if(!f.isDirectory()){
                f.mkdirs();
            }

            String[] paras={"","semeion","1"};
            DataSets.problemPath="C:/Data/UCI Problems/";
            DataSets.resultsPath=DataSets.dropboxPath+"Results/UCIResults/";
            File file =new File("C:\\Users\\ajb\\Dropbox\\Results\\UCIResults");
            paras[0]="RotFCV";
            paras[2]="1";
            July2017Experiments.singleClassifierAndFoldSingleDataSet(paras);            
            long t1=System.currentTimeMillis();
            for(int i=2;i<=11;i++){
                paras[2]=i+"";
                July2017Experiments.singleClassifierAndFoldSingleDataSet(paras);            
            }
            long t2=System.currentTimeMillis();
            paras[0]="EnhancedRotF";
            July2017Experiments.singleClassifierAndFoldSingleDataSet(paras);            
            long t3=System.currentTimeMillis();
            for(int i=2;i<=11;i++){
                paras[2]=i+"";
                July2017Experiments.singleClassifierAndFoldSingleDataSet(paras);            
            }
            long t4=System.currentTimeMillis();
            System.out.println("Standard = "+(t2-t1)+", Enhanced = "+(t4-t3));
            
       }        
    }



}

