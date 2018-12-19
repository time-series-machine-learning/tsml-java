/*
To do: 

RandomForest: 
what exactly does EnhancedRandomForest do? Make it so that it can recover 
all thats necessary for rebuild from train (individual train preds and 
overall CV weight)

CAWPE: Add in final build to the buildClassifier, missing in original
    Result: Compare vs CAWPE. Continue with this as baseline
HESCAV1: Simple rule based to set folds as 10x for larger problems
    Result: Compare vs HESCAV1 
HESCAV2: Switch RandForest to use OOB error
    Result: Compare vs HESCAV1, generate timings locally.  
OK, works, but need to make it keep CVPredictions


CAWPE Development:
1. ADD MORE CLASSIFIERS
    Other ensembles: AdaBoost etc
2. Optimize existing:
    Find previous SMO code and run on all UCI with optimised window

All depreciated by James's changes

*/


package development;

import fileIO.OutFile;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.meta.RotationForest;
import vector_classifiers.CAWPE;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class HESCADevelopment {

    
    
    public static void testReadWrite(){
        Instances train = ClassifierTools.loadData("c:/tsc problems/ItalyPowerDemand/ItalyPowerDemand_TRAIN");
        Instances test = ClassifierTools.loadData("c:/tsc problems/ItalyPowerDemand/ItalyPowerDemand_TEST");
        
//        buildAndWriteFullIndividualTrainTestResults(train, test, "hescatest/", "ItalyPowerDemand", "htest", 0, null);
        CAWPE h = new CAWPE();
        h.setRandSeed(0);
        h.setDebug(true);
        h.setResultsFileLocationParameters(DataSets.resultsPath,"ItalyPowerDemand", 0);
        h.setBuildIndividualsFromResultsFiles(true);
        
    }
    
    
    /*  Work out how long each hesca component takes in default config
    with range of series lengths
    */    
    
    
     public static void timingExperimentInTrainSize() throws Exception{
        OutFile out=new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\HESCATest\\HescaBasicTimingTrainSizeV123.csv");
          CAWPE hesca=new CAWPE();
        Classifier[] classifierNames=hesca.getClassifiers();
        for(int c = 0; c < classifierNames.length; c++)
            out.writeString(","+classifierNames[c].getClass().getName());
        out.writeString(", HESCA\n");
        MatrixProfileExperiments.seriesLength=100;
        for(int s=250;s<=5000;s+=250){
            MatrixProfileExperiments.casesPerClass=new int[]{s/2,s/2};
            MatrixProfileExperiments.trainProp=0.2;
            Instances d= SimulationExperiments.simulateData("Interval",0);
            out.writeString("\n"+s+",");
            System.out.println(" s ="+s+"   ");
            hesca=new CAWPE();
            long start=System.nanoTime();
            hesca.buildClassifier(d);
            long end=System.nanoTime();
            out.writeString(","+(end-start));
            System.out.println(hesca.getClass().getName()+"::"+((end-start)/1000000000.0)+"  ");
            hesca=new CAWPE();
            start=System.nanoTime();
            hesca.buildClassifier(d);
            end=System.nanoTime();
            out.writeString(","+(end-start));
            System.out.println(hesca.getClass().getName()+"::"+((end-start)/1000000000.0)+"  ");
            hesca=new CAWPE();
            start=System.nanoTime();
            hesca.buildClassifier(d);
            end=System.nanoTime();
            out.writeString(","+(end-start));

            System.out.println(hesca.getClass().getName()+"::"+((end-start)/1000000000.0)+"  ");
        }
    } 
    public static void timingExperimentInSeriesLength() throws Exception{
        OutFile out=new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\SimulationExperiments\\HESCATest\\HescaBasicTimingSeriesLengthV123.csv");
          CAWPE hesca=new CAWPE();
        Classifier[] classifierNames=hesca.getClassifiers();
        for(int c = 0; c < classifierNames.length; c++)
            out.writeString(","+classifierNames[c].getClass().getName());
        out.writeString(", HESCA\n");
        for(int s=100;s<=1000;s+=100){
            MatrixProfileExperiments.seriesLength=s;
            Instances d= SimulationExperiments.simulateData("Interval",0);
            out.writeString("\n"+s+",");
            System.out.println(" s ="+s+"   ");
            hesca=new CAWPE();
            long start=System.nanoTime();
            hesca.buildClassifier(d);
            long end=System.nanoTime();
            out.writeString(","+(end-start));
            System.out.println(hesca.getClass().getName()+"::"+((end-start)/1000000000.0)+"  ");
            hesca=new CAWPE();
            start=System.nanoTime();
            hesca.buildClassifier(d);
            end=System.nanoTime();
            out.writeString(","+(end-start));
            System.out.println(hesca.getClass().getName()+"::"+((end-start)/1000000000.0)+"  ");
            hesca=new CAWPE();
            start=System.nanoTime();
            hesca.buildClassifier(d);
            end=System.nanoTime();
            out.writeString(","+(end-start));

            System.out.println(hesca.getClass().getName()+"::"+((end-start)/1000000000.0)+"  ");
        }
    }
    
/** 
 * Test speed of doing a CV vs using OOB error
 * @throws Exception 
 */
    public static void randomForestTimes() throws Exception{
        OutFile out=new OutFile("C:/Temp/randomForest.csv");
          CAWPE hesca=new CAWPE();
        out.writeString(", CV,OOB\n");
        for(int s=100;s<=1000;s+=100){
            MatrixProfileExperiments.seriesLength=s;
            Instances d= SimulationExperiments.simulateData("Interval",0);
            out.writeString("\n"+s+",");
            System.out.println(" s ="+s+"   ");
            RandomForest rf=new RandomForest();
            rf.setNumTrees(500);
            long start=System.nanoTime();
            //hesca.crossValidate(rf, d);
            rf.buildClassifier(d);
            long end=System.nanoTime();
            out.writeString(","+(end-start));
            System.out.println(" RandF CV time  ="+((end-start)/1000000000.0));
            rf=new RandomForest();
            rf.setNumTrees(500);
            start=System.nanoTime();
            rf.buildClassifier(d);
            double oob=1-rf.measureOutOfBagError();
            end=System.nanoTime();
            out.writeString(","+(end-start));
            System.out.println("Rand F oob time "+((end-start)/1000000000.0)+"  ");
            hesca=new CAWPE();
            start=System.nanoTime();
            hesca.buildClassifier(d);
            end=System.nanoTime();
            out.writeString(","+(end-start));

            System.out.println("  hesca time "+((end-start)/1000000000.0)+"  ");
        }
    }
    
/** 
 * Test speed for different numbers of classifiers
 * @throws Exception 
 */
    public static void rotationForestTimesForL() throws Exception{
        OutFile out=new OutFile("C:/Temp/EnhancedRotFTest.csv");
        RotationForest[] rotf=new RotationForest[2];
        RotationForestLimitedAttributes[] rotf2=new RotationForestLimitedAttributes[5];
        for(int i=0;i<rotf.length;i++){
            rotf[i]=new RotationForest();
            rotf[i].setNumIterations(10+i*40);
            rotf2[i]=new RotationForestLimitedAttributes();
            rotf2[i].setNumIterations(10+i*40);
        }
        out.writeString(", 10,20,30,40,50\n");
        for(int s=1000;s<=3000;s+=200){
            MatrixProfileExperiments.seriesLength=s;
            Instances d= SimulationExperiments.simulateData("Interval",0);
            Instances[] split=InstanceTools.resampleInstances(d, 0,0.2);
            out.writeString("\n"+s+",");
            System.out.println(" s ="+s+"   ");
            for(RotationForest rf:rotf){
                long start=System.nanoTime();
                double acc=ClassifierTools.singleTrainTestSplitAccuracy(rf, split[0],split[1]);
                long end=System.nanoTime();
                out.writeString(","+(end-start)+","+acc);
                System.out.println(" RandF time  ="+((end-start)/1000000000.0)+" acc= "+acc);
            }
            out.writeString(",");            
            for(RotationForest rf:rotf2){
                long start=System.nanoTime();
                double acc=ClassifierTools.singleTrainTestSplitAccuracy(rf, split[0],split[1]);
                long end=System.nanoTime();
                out.writeString(","+(end-start)+","+acc);
                System.out.println(" RotF ENHANCED time  ="+((end-start)/1000000000.0)+" acc= "+acc);
            }
        }
/*        out=new OutFile("C:/Temp/rotForestNumTreesNumCases.csv");
        out.writeString(", 10,20,30,40,50\n");
        MatrixProfileExperiments.seriesLength=100;
        for(int s=250;s<=2500;s+=250){
             MatrixProfileExperiments.casesPerClass=new int[]{s/2,s/2};
            MatrixProfileExperiments.trainProp=0.2;
            Instances d= MatrixProfileExperiments.simulateData("Interval",0);
            Instances[] split=InstanceTools.resampleInstances(d, 0,0.5);
            out.writeString("\n"+s+",");
            System.out.println(" s ="+s+"   ");
            for(RotationForest rf:rotf){
                long start=System.nanoTime();
                double acc=ClassifierTools.singleTrainTestSplitAccuracy(rf, split[0],split[1]);
                long end=System.nanoTime();
                out.writeString(","+(end-start)+","+acc);
                System.out.println(" RandF time  ="+((end-start)/1000000000.0)+" acc= "+acc);
            }
                out.writeString("\n");
        }
      */  
    }
    

    public static void rotationForestTimesForNosGroups() throws Exception{
        OutFile out=new OutFile("C:/Temp/rotForestNumGroupsSeriesLength.csv");
        RotationForest[] rotf=new RotationForest[5];
        for(int i=0;i<rotf.length;i++){
            rotf[i]=new RotationForest();
            rotf[i].setNumIterations(10+i*10);
        }
        out.writeString(", 10,20,30,40,50\n");
        for(int s=100;s<=1000;s+=100){
            MatrixProfileExperiments.seriesLength=s;
            Instances d= SimulationExperiments.simulateData("Interval",0);
            Instances[] split=InstanceTools.resampleInstances(d, 0,0.5);
            out.writeString("\n"+s+",");
            System.out.println(" s ="+s+"   ");
            for(RotationForest rf:rotf){
                long start=System.nanoTime();
                double acc=ClassifierTools.singleTrainTestSplitAccuracy(rf, split[0],split[1]);
                long end=System.nanoTime();
                out.writeString(","+(end-start)+","+acc);
                System.out.println(" RandF time  ="+((end-start)/1000000000.0)+" acc= "+acc);
            }
                out.writeString("\n");
        }
        out=new OutFile("C:/Temp/rotForestNumTreesNumCases.csv");
        out.writeString(", 10,20,30,40,50\n");
        MatrixProfileExperiments.seriesLength=100;
        for(int s=250;s<=2500;s+=250){
             MatrixProfileExperiments.casesPerClass=new int[]{s/2,s/2};
            MatrixProfileExperiments.trainProp=0.2;
            Instances d= SimulationExperiments.simulateData("Interval",0);
            Instances[] split=InstanceTools.resampleInstances(d, 0,0.5);
            out.writeString("\n"+s+",");
            System.out.println(" s ="+s+"   ");
            for(RotationForest rf:rotf){
                long start=System.nanoTime();
                double acc=ClassifierTools.singleTrainTestSplitAccuracy(rf, split[0],split[1]);
                long end=System.nanoTime();
                out.writeString(","+(end-start)+","+acc);
                System.out.println(" RandF time  ="+((end-start)/1000000000.0)+" acc= "+acc);
            }
                out.writeString("\n");
        }
        
    }
    
    
    
    public static void main(String[] args) throws Exception {

rotationForestTimesForL();
//        rotationForestTimes();
        //timingExperimentInTrainSize();
        
//        randomForestTimes();
//        RotationForest rf;
        
//        timingExperimentWithCV();
    }
/*    
    public static class HESCAV1 extends CAWPE{
        public static int FOLDS1=20;
        public static int FOLDS2=10;
    @Override
    public void buildClassifier(Instances input) throws Exception{
        if(this.transform==null){
            this.train = input;
        }else{
            this.train = transform.process(input);
        }
        
        int correct;  
        this.individualCvPreds = new double[this.classifiers.length][this.train.numInstances()];
        this.individualCvAccs = new double[this.classifiers.length];
            
        for(int c = 0; c < this.classifiers.length; c++){
        {
                this.individualCvPreds[c] = crossValidate(this.classifiers[c],this.train);
                correct = 0;

                for(int i = 0; i < this.individualCvPreds[c].length; i++){
                    if(train.instance(i).classValue()==this.individualCvPreds[c][i]){
                        correct++;
                    }
                }
                this.individualCvAccs[c] = (double)correct/train.numInstances();
                classifiers[c].buildClassifier(train);
            }
        }
            
        this.ensembleCvPreds = new double[train.numInstances()];
        double actual, pred;
        double bsfWeight;
        correct = 0;
        ArrayList<Double> bsfClassVals;
        double[] weightByClass;
        for(int i = 0; i < train.numInstances(); i++){
            actual = train.instance(i).classValue();
            bsfClassVals = null;
            bsfWeight = -1;
            weightByClass = new double[train.numClasses()];

            for(int m = 0; m < classifiers.length; m++){
                weightByClass[(int)individualCvPreds[m][i]]+=this.individualCvAccs[m];
                
                if(weightByClass[(int)individualCvPreds[m][i]] > bsfWeight){
                    bsfWeight = weightByClass[(int)individualCvPreds[m][i]];
                    bsfClassVals = new ArrayList<>();
                    bsfClassVals.add(individualCvPreds[m][i]);
                }else if(weightByClass[(int)individualCvPreds[m][i]] == bsfWeight){
                    bsfClassVals.add(individualCvPreds[m][i]);
                }
            }
            
            if(bsfClassVals.size()>1){
                pred = bsfClassVals.get(new Random().nextInt(bsfClassVals.size()));
            }else{
                pred = bsfClassVals.get(0);
            }
            
            if(pred==actual){
                correct++;
            }
            this.ensembleCvPreds[i]=pred;
        }
        this.ensembleCvAcc = (double)correct/train.numInstances();
/*        if(this.writeTraining){
            StringBuilder output = new StringBuilder();

            String hescaIdentifier = "HESCAV1";
            if(this.transform!=null){
                hescaIdentifier = "HESCA_"+this.transform.getClass().getSimpleName();
            }
           
            output.append(input.relationName()).append(","+hescaIdentifier+",train\n");
            output.append(this.getParameters()).append("\n");
            output.append(this.getEnsembleCvAcc()).append("\n");

            for(int i = 0; i < train.numInstances(); i++){
                output.append(train.instance(i).classValue()).append(",").append(ensembleCvPreds[i]).append("\n");
            }

            new File(this.outputTrainingPathAndFile).getParentFile().mkdirs();
            FileWriter fullTrain = new FileWriter(this.outputTrainingPathAndFile);
            fullTrain.append(output);
            fullTrain.close();
        }        
        
    }

    
    public static int findNumFolds(Instances train){
        int numFolds = train.numInstances();
        if(train.numInstances()>=300)
            numFolds=FOLDS2;
        else if(train.numInstances()>=200 && train.numAttributes()>=200)
            numFolds=FOLDS2;
        else if(train.numAttributes()>=600)
            numFolds=FOLDS2;
        else if (train.numInstances()>=100) 
            numFolds=FOLDS1;
        return numFolds;
    }
    public double[] crossValidate(Classifier classifier, Instances train) throws Exception{

        int numFolds=findNumFolds(train);
            r = new Random();
               
        Random r = null;
/*        if(this.setSeed){
            r = new Random(this.seed);
        }else{
            r = new Random();
        }

        ArrayList<Instances> folds = new ArrayList<>();
        ArrayList<ArrayList<Integer>> foldIndexing = new ArrayList<>();
        
        for(int i = 0; i < numFolds; i++){
            folds.add(new Instances(train,0));
            foldIndexing.add(new ArrayList<>());
        }
        
        ArrayList<Integer> instanceIds = new ArrayList<>();
        for(int i = 0; i < train.numInstances(); i++){
            instanceIds.add(i);
        }
        Collections.shuffle(instanceIds, r);
        
        ArrayList<Instances> byClass = new ArrayList<>();
        ArrayList<ArrayList<Integer>> byClassIndices = new ArrayList<>();
        for(int i = 0; i < train.numClasses(); i++){
            byClass.add(new Instances(train,0));
            byClassIndices.add(new ArrayList<>());
        }
        
        int thisInstanceId;
        double thisClassVal;
        for(int i = 0; i < train.numInstances(); i++){
            thisInstanceId = instanceIds.get(i);
            thisClassVal = train.instance(thisInstanceId).classValue();
            
            byClass.get((int)thisClassVal).add(train.instance(thisInstanceId));
            byClassIndices.get((int)thisClassVal).add(thisInstanceId);
        }

         // now stratify        
        Instances strat = new Instances(train,0);
        ArrayList<Integer> stratIndices = new ArrayList<>();
        int stratCount = 0;
        int[] classCounters = new int[train.numClasses()];
        
        while(stratCount < train.numInstances()){
            
            for(int c = 0; c < train.numClasses(); c++){
                if(classCounters[c] < byClass.get(c).size()){
                    strat.add(byClass.get(c).instance(classCounters[c]));
                    stratIndices.add(byClassIndices.get(c).get(classCounters[c]));
                    classCounters[c]++;
                    stratCount++;
                }
            }
        }
        

        train = strat;
        instanceIds = stratIndices;
       
        double foldSize = (double)train.numInstances()/numFolds;
        
        double thisSum = 0;
        double lastSum = 0;
        int floor;
        int foldSum = 0;
        

        int currentStart = 0;
        for(int f = 0; f < numFolds; f++){

            
            thisSum = lastSum+foldSize+0.000000000001;  
// to try and avoid double imprecision errors (shouldn't ever be big enough to effect folds when double imprecision isn't an issue)
            floor = (int)thisSum;
            
            if(f==numFolds-1){
                floor = train.numInstances(); // to make sure all instances are allocated in case of double imprecision causing one to go missing
            }
            
            for(int i = currentStart; i < floor; i++){
                folds.get(f).add(train.instance(i));
                foldIndexing.get(f).add(instanceIds.get(i));
            }

            foldSum+=(floor-currentStart);
            currentStart = floor;
            lastSum = thisSum;
        }
        
        if(foldSum!=train.numInstances()){
            throw new Exception("Error! Some instances got lost file creating folds (maybe a double precision bug). Training instances contains "+train.numInstances()+", but the sum of the training folds is "+foldSum);
        }
        

        Instances trainLoocv;
        Instances testLoocv;
        
        double pred, actual;
        double[] predictions = new double[train.numInstances()];
        
        int correct = 0;
        Instances temp; // had to add in redundant instance storage so we don't keep killing the base set of Instances by mistake
        
        for(int testFold = 0; testFold < numFolds; testFold++){
            
            trainLoocv = null;
            testLoocv = new Instances(folds.get(testFold));
            
            for(int f = 0; f < numFolds; f++){
                if(f==testFold){
                    continue;
                }
                temp = new Instances(folds.get(f));
                if(trainLoocv==null){
                    trainLoocv = temp;
                }else{
                    trainLoocv.addAll(temp);
                }
            }
            
            classifier.buildClassifier(trainLoocv);
            for(int i = 0; i < testLoocv.numInstances(); i++){
                pred = classifier.classifyInstance(testLoocv.instance(i));
                actual = testLoocv.instance(i).classValue();
                predictions[foldIndexing.get(testFold).get(i)] = pred;
                if(pred==actual){
                    correct++;
                }
            }
        }
        
        return predictions;
    }

    
    
    }

    public static class HESCAV2 extends CAWPE{
        public static int FOLDS1=20;
        public static int FOLDS2=10;
    @Override
    public void buildClassifier(Instances input) throws Exception{
        if(this.transform==null){
            this.train = input;
        }else{
            this.train = transform.process(input);
        }
        
        int correct;  
        this.individualCvPreds = new double[this.classifiers.length][this.train.numInstances()];
        this.individualCvAccs = new double[this.classifiers.length];
            
        for(int c = 0; c < this.classifiers.length; c++){
//            if(classifiers[c] instanceof BaggedRandomForest){
// Implement after testing whether 10x fold is as accurate             
            if(classifiers[c] instanceof RandomForest){
                classifiers[c].buildClassifier(train);
//Individual CvPreds required!                
                individualCvAccs[c]=1-((RandomForest)classifiers[c]).measureOutOfBagError();                
            }
            else 
            {
                this.individualCvPreds[c] = crossValidate(this.classifiers[c],this.train);
                correct = 0;

                for(int i = 0; i < this.individualCvPreds[c].length; i++){
                    if(train.instance(i).classValue()==this.individualCvPreds[c][i]){
                        correct++;
                    }
                }
                this.individualCvAccs[c] = (double)correct/train.numInstances();
                classifiers[c].buildClassifier(train);
            }
        }
            
        this.ensembleCvPreds = new double[train.numInstances()];
        double actual, pred;
        double bsfWeight;
        correct = 0;
        ArrayList<Double> bsfClassVals;
        double[] weightByClass;
        for(int i = 0; i < train.numInstances(); i++){
            actual = train.instance(i).classValue();
            bsfClassVals = null;
            bsfWeight = -1;
            weightByClass = new double[train.numClasses()];

            for(int m = 0; m < classifiers.length; m++){
                weightByClass[(int)individualCvPreds[m][i]]+=this.individualCvAccs[m];
                
                if(weightByClass[(int)individualCvPreds[m][i]] > bsfWeight){
                    bsfWeight = weightByClass[(int)individualCvPreds[m][i]];
                    bsfClassVals = new ArrayList<>();
                    bsfClassVals.add(individualCvPreds[m][i]);
                }else if(weightByClass[(int)individualCvPreds[m][i]] == bsfWeight){
                    bsfClassVals.add(individualCvPreds[m][i]);
                }
            }
            
            if(bsfClassVals.size()>1){
                pred = bsfClassVals.get(new Random().nextInt(bsfClassVals.size()));
            }else{
                pred = bsfClassVals.get(0);
            }
            
            if(pred==actual){
                correct++;
            }
            this.ensembleCvPreds[i]=pred;
        }
        this.ensembleCvAcc = (double)correct/train.numInstances();
        if(this.writeTraining){
            StringBuilder output = new StringBuilder();

            String hescaIdentifier = "HESCAV1";
            if(this.transform!=null){
                hescaIdentifier = "HESCA_"+this.transform.getClass().getSimpleName();
            }
            
            output.append(input.relationName()).append(","+hescaIdentifier+",train\n");
            output.append(this.getParameters()).append("\n");
            output.append(this.getEnsembleCvAcc()).append("\n");

            for(int i = 0; i < train.numInstances(); i++){
                output.append(train.instance(i).classValue()).append(",").append(ensembleCvPreds[i]).append("\n");
            }

            new File(this.outputTrainingPathAndFile).getParentFile().mkdirs();
            FileWriter fullTrain = new FileWriter(this.outputTrainingPathAndFile);
            fullTrain.append(output);
            fullTrain.close();
        }        
                
    }

    
 
    public double[] crossValidate(Classifier classifier, Instances train) throws Exception{

        int numFolds=HESCAV1.findNumFolds(train);
               
        Random r = null;
        if(this.setSeed){
            r = new Random(this.seed);
        }else{
            r = new Random();
        }
        ArrayList<Instances> folds = new ArrayList<>();
        ArrayList<ArrayList<Integer>> foldIndexing = new ArrayList<>();
        
        for(int i = 0; i < numFolds; i++){
            folds.add(new Instances(train,0));
            foldIndexing.add(new ArrayList<>());
        }
        
        ArrayList<Integer> instanceIds = new ArrayList<>();
        for(int i = 0; i < train.numInstances(); i++){
            instanceIds.add(i);
        }
        Collections.shuffle(instanceIds, r);
        
        ArrayList<Instances> byClass = new ArrayList<>();
        ArrayList<ArrayList<Integer>> byClassIndices = new ArrayList<>();
        for(int i = 0; i < train.numClasses(); i++){
            byClass.add(new Instances(train,0));
            byClassIndices.add(new ArrayList<>());
        }
        
        int thisInstanceId;
        double thisClassVal;
        for(int i = 0; i < train.numInstances(); i++){
            thisInstanceId = instanceIds.get(i);
            thisClassVal = train.instance(thisInstanceId).classValue();
            
            byClass.get((int)thisClassVal).add(train.instance(thisInstanceId));
            byClassIndices.get((int)thisClassVal).add(thisInstanceId);
        }

         // now stratify        
        Instances strat = new Instances(train,0);
        ArrayList<Integer> stratIndices = new ArrayList<>();
        int stratCount = 0;
        int[] classCounters = new int[train.numClasses()];
        
        while(stratCount < train.numInstances()){
            
            for(int c = 0; c < train.numClasses(); c++){
                if(classCounters[c] < byClass.get(c).size()){
                    strat.add(byClass.get(c).instance(classCounters[c]));
                    stratIndices.add(byClassIndices.get(c).get(classCounters[c]));
                    classCounters[c]++;
                    stratCount++;
                }
            }
        }
        

        train = strat;
        instanceIds = stratIndices;
       
        double foldSize = (double)train.numInstances()/numFolds;
        
        double thisSum = 0;
        double lastSum = 0;
        int floor;
        int foldSum = 0;
        

        int currentStart = 0;
        for(int f = 0; f < numFolds; f++){

            
            thisSum = lastSum+foldSize+0.000000000001;  
// to try and avoid double imprecision errors (shouldn't ever be big enough to effect folds when double imprecision isn't an issue)
            floor = (int)thisSum;
            
            if(f==numFolds-1){
                floor = train.numInstances(); // to make sure all instances are allocated in case of double imprecision causing one to go missing
            }
            
            for(int i = currentStart; i < floor; i++){
                folds.get(f).add(train.instance(i));
                foldIndexing.get(f).add(instanceIds.get(i));
            }

            foldSum+=(floor-currentStart);
            currentStart = floor;
            lastSum = thisSum;
        }
        
        if(foldSum!=train.numInstances()){
            throw new Exception("Error! Some instances got lost file creating folds (maybe a double precision bug). Training instances contains "+train.numInstances()+", but the sum of the training folds is "+foldSum);
        }
        

        Instances trainLoocv;
        Instances testLoocv;
        
        double pred, actual;
        double[] predictions = new double[train.numInstances()];
        
        int correct = 0;
        Instances temp; // had to add in redundant instance storage so we don't keep killing the base set of Instances by mistake
        
        for(int testFold = 0; testFold < numFolds; testFold++){
            
            trainLoocv = null;
            testLoocv = new Instances(folds.get(testFold));
            
            for(int f = 0; f < numFolds; f++){
                if(f==testFold){
                    continue;
                }
                temp = new Instances(folds.get(f));
                if(trainLoocv==null){
                    trainLoocv = temp;
                }else{
                    trainLoocv.addAll(temp);
                }
            }
            
            classifier.buildClassifier(trainLoocv);
            for(int i = 0; i < testLoocv.numInstances(); i++){
                pred = classifier.classifyInstance(testLoocv.instance(i));
                actual = testLoocv.instance(i).classValue();
                predictions[foldIndexing.get(testFold).get(i)] = pred;
                if(pred==actual){
                    correct++;
                }
            }
        }
        
        return predictions;
    }

    
    
    }


    public static class HESCAV3 extends HESCAV2{
    public void buildClassifier(Instances input) throws Exception{
        if(this.transform==null){
            this.train = input;
        }else{
            this.train = transform.process(input);
        }
        
        int correct;  
        this.individualCvPreds = new double[this.classifiers.length][this.train.numInstances()];
        this.individualCvAccs = new double[this.classifiers.length];
            
        for(int c = 0; c < this.classifiers.length; c++){
//            if(classifiers[c] instanceof BaggedRandomForest){
// Implement after testing whether 10x fold is as accurate             
            if(classifiers[c] instanceof RandomForest){
                classifiers[c].buildClassifier(train);
                individualCvAccs[c]=1-((RandomForest)classifiers[c]).measureOutOfBagError();                
            }
            else 
            {
                this.individualCvPreds[c] = crossValidate(this.classifiers[c],this.train);
                correct = 0;

                for(int i = 0; i < this.individualCvPreds[c].length; i++){
                    if(train.instance(i).classValue()==this.individualCvPreds[c][i]){
                        correct++;
                    }
                }
                this.individualCvAccs[c] = (double)correct/train.numInstances();
                classifiers[c].buildClassifier(train);
            }
        }
            
        this.ensembleCvPreds = new double[train.numInstances()];
        double actual, pred;
        double bsfWeight;
        correct = 0;
        ArrayList<Double> bsfClassVals;
        double[] weightByClass;
        for(int i = 0; i < train.numInstances(); i++){
            actual = train.instance(i).classValue();
            bsfClassVals = null;
            bsfWeight = -1;
            weightByClass = new double[train.numClasses()];

            for(int m = 0; m < classifiers.length; m++){
                weightByClass[(int)individualCvPreds[m][i]]+=this.individualCvAccs[m];
                
                if(weightByClass[(int)individualCvPreds[m][i]] > bsfWeight){
                    bsfWeight = weightByClass[(int)individualCvPreds[m][i]];
                    bsfClassVals = new ArrayList<>();
                    bsfClassVals.add(individualCvPreds[m][i]);
                }else if(weightByClass[(int)individualCvPreds[m][i]] == bsfWeight){
                    bsfClassVals.add(individualCvPreds[m][i]);
                }
            }
            
            if(bsfClassVals.size()>1){
                pred = bsfClassVals.get(new Random().nextInt(bsfClassVals.size()));
            }else{
                pred = bsfClassVals.get(0);
            }
            
            if(pred==actual){
                correct++;
            }
            this.ensembleCvPreds[i]=pred;
        }
        this.ensembleCvAcc = (double)correct/train.numInstances();
        if(this.writeTraining){
            StringBuilder output = new StringBuilder();

            String hescaIdentifier = "HESCAV1";
            if(this.transform!=null){
                hescaIdentifier = "HESCA_"+this.transform.getClass().getSimpleName();
            }
            
            output.append(input.relationName()).append(","+hescaIdentifier+",train\n");
            output.append(this.getParameters()).append("\n");
            output.append(this.getEnsembleCvAcc()).append("\n");

            for(int i = 0; i < train.numInstances(); i++){
                output.append(train.instance(i).classValue()).append(",").append(ensembleCvPreds[i]).append("\n");
            }

            new File(this.outputTrainingPathAndFile).getParentFile().mkdirs();
            FileWriter fullTrain = new FileWriter(this.outputTrainingPathAndFile);
            fullTrain.append(output);
            fullTrain.close();
        }        
                
    }

 
    public double[] crossValidate(Instances train) throws Exception{

        int numFolds=HESCAV1.findNumFolds(train);
               
        Random r = null;
        if(this.setSeed){
            r = new Random(this.seed);
        }else{
            r = new Random();
        }
        ArrayList<Instances> folds = new ArrayList<>();
        ArrayList<ArrayList<Integer>> foldIndexing = new ArrayList<>();
        
        for(int i = 0; i < numFolds; i++){
            folds.add(new Instances(train,0));
            foldIndexing.add(new ArrayList<>());
        }
        
        ArrayList<Integer> instanceIds = new ArrayList<>();
        for(int i = 0; i < train.numInstances(); i++){
            instanceIds.add(i);
        }
        Collections.shuffle(instanceIds, r);
        
        ArrayList<Instances> byClass = new ArrayList<>();
        ArrayList<ArrayList<Integer>> byClassIndices = new ArrayList<>();
        for(int i = 0; i < train.numClasses(); i++){
            byClass.add(new Instances(train,0));
            byClassIndices.add(new ArrayList<>());
        }
        
        int thisInstanceId;
        double thisClassVal;
        for(int i = 0; i < train.numInstances(); i++){
            thisInstanceId = instanceIds.get(i);
            thisClassVal = train.instance(thisInstanceId).classValue();
            
            byClass.get((int)thisClassVal).add(train.instance(thisInstanceId));
            byClassIndices.get((int)thisClassVal).add(thisInstanceId);
        }

         // now stratify        
        Instances strat = new Instances(train,0);
        ArrayList<Integer> stratIndices = new ArrayList<>();
        int stratCount = 0;
        int[] classCounters = new int[train.numClasses()];
        
        while(stratCount < train.numInstances()){
            for(int c = 0; c < train.numClasses(); c++){
                if(classCounters[c] < byClass.get(c).size()){
                    strat.add(byClass.get(c).instance(classCounters[c]));
                    stratIndices.add(byClassIndices.get(c).get(classCounters[c]));
                    classCounters[c]++;
                    stratCount++;
                }
            }
        }
        train = strat;
        instanceIds = stratIndices;
        double foldSize = (double)train.numInstances()/numFolds;
        
        double thisSum = 0;
        double lastSum = 0;
        int floor;
        int foldSum = 0;
        

        int currentStart = 0;
        for(int f = 0; f < numFolds; f++){
            thisSum = lastSum+foldSize+0.000000000001;  
// to try and avoid double imprecision errors (shouldn't ever be big enough to effect folds when double imprecision isn't an issue)
            floor = (int)thisSum;
            if(f==numFolds-1){
                floor = train.numInstances(); // to make sure all instances are allocated in case of double imprecision causing one to go missing
            }
            for(int i = currentStart; i < floor; i++){
                folds.get(f).add(train.instance(i));
                foldIndexing.get(f).add(instanceIds.get(i));
            }
            foldSum+=(floor-currentStart);
            currentStart = floor;
            lastSum = thisSum;
        }
        
        if(foldSum!=train.numInstances()){
            throw new Exception("Error! Some instances got lost file creating folds (maybe a double precision bug). Training instances contains "+train.numInstances()+", but the sum of the training folds is "+foldSum);
        }
        Instances trainFold,testFold;
        double pred, actual;
        double[] predictions = new double[train.numInstances()];
        
        int correct = 0;
        Instances temp; 
// had to add in redundant instance storage so we don't keep killing the base 
// set of Instances by mistake
        for(int fold = 0; fold < numFolds; fold++){
            trainFold = null;
            testFold = new Instances(folds.get(fold));
            for(int f = 0; f < numFolds; f++){
                if(f==fold){
                    continue;
                }
                temp = new Instances(folds.get(f));
                if(trainFold==null){
                    trainFold = temp;
                }else{
                    trainFold.addAll(temp);
                }
            }
//Build all classifiers now, rather than rebuild each time
            for(Classifier cls:classifiers){
                cls.buildClassifier(trainFold);
                for(int i = 0; i < testFold.numInstances(); i++){
                    pred = cls.classifyInstance(testFold.instance(i));
                    actual = testFold.instance(i).classValue();
                    predictions[foldIndexing.get(fold).get(i)] = pred;
                    if(pred==actual){
                        correct++;
                    }
                }
          }
        }
        
        return predictions;
    }

    

        
    }
*/
    
}


