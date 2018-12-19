/*
Time Series Bag of Features (TSBF): Baydogan

Time series classification with a bag-of-features (TSBF) algorithm.
series length =m, num series =n

PARAMETERS
    minimum interval length: wmin=5;   
    bin size for codebook generation in phase 2:   binsize=10      
VARIABLES
    number of intervals per subseries 
            numIntervals=(int)((zLevel*seriesLength)/wmin);
    number of subseries:    
            numSub=  (seriesLength/wmin)-numIntervals;

1. Subsequences are sampled and partitioned into intervals for feature extraction.

    number of subseries numSub= floor(m/wmin)-d
    each subseries is of random length ls
    each subseries is split into d segments
    mean, variance and slope is extracted for each segment

For i=1 to number of  subsequences
    select start and end point s1 and s2
    for each time series t in T
                    generate intervals on t_s1 and t_s2
                    generate features (mean, std dev and slope) from intervals
                    add to new features for t
 

	nos of features per sub series=3*d+4
        nos features per series = numSub*(3*d+4)

This forms a new data set that is identical to TSF except for the global features.

2. "Each subsequence feature set is labeled with the class of the time series and 
each time series forms the bag." 
I think it works by building a random forest on the labelled transformed subseries and use the 
class probability estimates from the forest. 



3. A classifier generates class probability estimates. 

4. Histograms of the class probability estimates are generated (and concatenated) to summarize the subsequence
information. 

5. Global features are added. 

6. A final classifier is then trained on the new representation to assign each time series.
 */
package timeseriesweka.classifiers;

import development.DataSets;
import fileIO.OutFile;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;
import timeseriesweka.classifiers.TSF.FeatureSet;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import vector_classifiers.TunedRandomForest;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;

/**
 *
 * @author ajb
 * 
 * PARAMETERS:
 *      zLevel: minimum subsequence length factors
 *      wmin:
 * ARGUMENTS
 * 
 */
public class TSBF extends AbstractClassifierWithTrainingData implements ParameterSplittable{
//Paras
    
//<editor-fold defaultstate="collapsed" desc="results reported in PAMI paper">        
    static double[] reportedResults={0.245,
0.287,
0.009,
0.336,
0.262,
0.004,
0.278,
0.259,
0.263,
0.126,
0.183,
0.234,
0.051,
0.090,
0.209,
0.080,
0.011,
0.488,
0.603,
0.096,
0.257,
0.262,
0.037,
0.269,
0.135,
0.138,
0.130,
0.090,
0.329,
0.175,
0.196,
0.022,
0.075,
0.034,
0.008,
0.020,
0.046,
0.001,
0.164,
0.249,
0.217,
0.004,
0.302,
0.149};
      //</editor-fold>  
    
//<editor-fold defaultstate="collapsed" desc="problems used in PAMI paper">   
    static String[] problems={"Adiac",
"Beef",
"CBF",
"ChlorineConcentration",
"CinCECGtorso",
"Coffee",
"CricketX",
"CricketY",
"CricketZ",
"DiatomSizeReduction",
"ECGFiveDays",
"FaceAll",
"FaceFour",
"FacesUCR",
"FiftyWords",
"Fish",
"GunPoint",
"Haptics",
"InlineSkate",
"ItalyPowerDemand",
"Lightning2",
"Lightning7",
"Mallat",
"MedicalImages",
"MoteStrain",
"NonInvasiveFatalECGThorax1",
"NonInvasiveFatalECGThorax2",
"OliveOil",
"OSULeaf",
"SonyAIBORobotSurface1",
"SonyAIBORobotSurface2",
"StarLightCurves",
"SwedishLeaf",
"Symbols",
"SyntheticControl",
"Trace",
"TwoLeadECG",
"TwoPatterns",
"UWaveGestureLibraryX",
"UWaveGestureLibraryY",
"UWaveGestureLibraryZ",
"Wafer",
"WordSynonyms",
"Yoga"};
      //</editor-fold>  
    
public static void recreatePublishedResults() throws Exception{
        OutFile of=new OutFile(DataSets.resultsPath+"RecreateTSBF.csv");
        System.out.println("problem,published,recreated");
        double meanDiff=0;
        int publishedBetter=0;
    for(int i=0;i<problems.length;i++){
        Instances train = ClassifierTools.loadData(DataSets.problemPath+problems[i]+"/"+problems[i]+"_TRAIN");
        Instances test = ClassifierTools.loadData(DataSets.problemPath+problems[i]+"/"+problems[i]+"_TEST");
        TSBF tsbf=new TSBF();
        tsbf.searchParameters(true);
        double a=ClassifierTools.singleTrainTestSplitAccuracy(tsbf, train, test);
        System.out.println(problems[i]+","+reportedResults[i]+","+(1-a));
        of.writeLine(problems[i]+","+reportedResults[i]+","+(1-a));
        meanDiff+=reportedResults[i]-(1-a);
        if(reportedResults[i]<(1-a))
            publishedBetter++;
    }
    System.out.println("Mean diff ="+meanDiff/problems.length+" Published better ="+publishedBetter);
        of.writeLine(",,,,Mean diff ="+meanDiff/problems.length+" Published better ="+publishedBetter);
}          

public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    
    result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
    result.setValue(TechnicalInformation.Field.AUTHOR, "M. Baydogan, G. Runger and E. Tuv");
    result.setValue(TechnicalInformation.Field.YEAR, "2013");
    result.setValue(TechnicalInformation.Field.TITLE, "A Bag-Of-Features Framework to Classify Time Series");
    result.setValue(TechnicalInformation.Field.JOURNAL, "IEEE Trans. PAMI");
    result.setValue(TechnicalInformation.Field.VOLUME, "35");
    result.setValue(TechnicalInformation.Field.NUMBER, "11");
    result.setValue(TechnicalInformation.Field.PAGES, "2796-2802");
    
    return result;
  }

    
    
    int minIntervalLength=5;   
    int numBins=10;      //bin size for codebook generation   
    int numReps=10;
    double oobError;
    static double[] zLevels={0.1,0.25,0.5,0.75}; //minimum subsequence length factors (z) to be evaluated
    double z=zLevels[0];
    int folds=10;
//Variables, dont need to be global, can be local to buildClassifier
    
    int seriesLength;   //data specific       
    int numIntervals;   //nos intervals per sub series=(int)((zLevel*seriesLength)/minIntervalLength);  
    int numSubSeries;         //nos subseries =  (int)(seriesLength/minIntervalLength)-numIntervals;
    int minSubLength;   // min series length = zlevel*seriesLength
    int numOfTreeStep=50; //step size for tree building process
    boolean paramSearch=true;
    double trainAcc;
    boolean stepWise=true;
    int[][] subSeries;
    int[][][] intervals;
    RandomForest subseriesRandomForest;
    RandomForest finalRandForest;

    Instances first;
    Random rand= new Random();
    static double TOLERANCE =0.05;
    public void seedRandom(int s){
        rand=new Random(s);
    }
    public void searchParameters(boolean b){
        paramSearch=b;
    }
    public void setZLevel(double zLevel){ z=zLevel;}
    public void setParametersFromIndex(int x){z=zLevels[x-1];}
    public String getParas(){ return z+"";}
    public double getAcc(){ return trainAcc;}
    @Override
    public void setParamSearch(boolean b){paramSearch =b;} 
    
    Instances formatIntervalInstances(Instances data){

        //3 stats for whole subseries, start and end point, 3 stats per interval
        int numFeatures=(3+2+3*numIntervals); 
        //Set up instances size and format. 
        FastVector atts=new FastVector();
        String name;
        for(int j=0;j<numFeatures;j++){
                name = "F"+j;
                atts.addElement(new Attribute(name));
        }
        //Get the class values as a fast vector			
        Attribute target =data.attribute(data.classIndex());
       FastVector vals=new FastVector(target.numValues());
        for(int j=0;j<target.numValues();j++)
                vals.addElement(target.value(j));
        atts.addElement(new Attribute(data.attribute(data.classIndex()).name(),vals));
//create blank instances with the correct class value                
        Instances result = new Instances("SubsequenceIntervals",atts,data.numInstances());
        result.setClassIndex(result.numAttributes()-1);
        for(int i=0;i<data.numInstances();i++){
            double cval=data.instance(i).classValue();
            for(int j=0;j<numSubSeries;j++){
                DenseInstance in=new DenseInstance(result.numAttributes());
                in.setValue(result.numAttributes()-1,cval);
                result.add(in);
            }
        }
        return result;
    }

    Instances formatProbabilityInstances(double[][] probs,Instances data){
        int numClasses=data.numClasses();
        int numFeatures=(numClasses-1)*numSubSeries;
        //Set up instances size and format. 
        FastVector atts=new FastVector();
        String name;
        for(int j=0;j<numFeatures;j++){
                name = "ProbFeature"+j;
                atts.addElement(new Attribute(name));
        }
        //Get the class values as a fast vector			
        Attribute target =data.attribute(data.classIndex());
       FastVector vals=new FastVector(target.numValues());
        for(int j=0;j<target.numValues();j++)
                vals.addElement(target.value(j));
        atts.addElement(new Attribute(data.attribute(data.classIndex()).name(),vals));
//create blank instances with the correct class value                
        Instances result = new Instances("SubsequenceIntervals",atts,data.numInstances());
        result.setClassIndex(result.numAttributes()-1);
        for(int i=0;i<data.numInstances();i++){
            double cval=data.instance(i).classValue();
            DenseInstance in=new DenseInstance(result.numAttributes());
            in.setValue(result.numAttributes()-1,cval);
            int pos=0;
            for(int j=0;j<numSubSeries;j++){
                for(int k=0;k<numClasses-1;k++)
                    in.setValue(pos++, probs[j+numSubSeries*i][k]);
            }           
            result.add(in);
        }
        return result;
    }
    

//count indexes i: instance, j = class count, k= bin
    Instances formatFrequencyBinInstances(int[][][] counts,double[][] classProbs,Instances data){
        int numClasses=data.numClasses();
        int numFeatures=numBins*(numClasses-1)+numClasses;
        //Set up instances size and format. 
        FastVector atts=new FastVector();
        String name;
        for(int j=0;j<numFeatures;j++){
                name = "FreqBinFeature"+j;
                atts.addElement(new Attribute(name));
        }
                
        //Get the class values as a fast vector			
        Attribute target =data.attribute(data.classIndex());
       FastVector vals=new FastVector(target.numValues());
        for(int j=0;j<target.numValues();j++)
                vals.addElement(target.value(j));
        atts.addElement(new Attribute(data.attribute(data.classIndex()).name(),vals));
//create blank instances with the correct class value                
        Instances result = new Instances("HistogramCounts",atts,data.numInstances());
        result.setClassIndex(result.numAttributes()-1);
        for(int i=0;i<data.numInstances();i++){
            double cval=data.instance(i).classValue();
            DenseInstance in=new DenseInstance(result.numAttributes());
            in.setValue(result.numAttributes()-1,cval);
            int pos=0;
//Set values here            
            for(int j=0;j<numClasses-1;j++){
                for(int k=0;k<numBins;k++)
                    in.setValue(pos++,counts[i][j][k]);
            }           
//            
           for(int j=0;j<numClasses;j++)
               in.setValue(pos++,classProbs[i][j]);
             
            result.add(in);
        }
        return result;
    }
    

    Classifier findIncrementalClassifier(Instances data) throws Exception{
 /*		iter=1
		while(iter<20&&cur_OOBerror<(1-tolerance)*prev_OOBerror){    
			prev_OOBerror=cur_OOBerror
			RFsubmid <- foreach(ntree=rep(noftree_step/nofthreads, nofthreads), .combine=combine, .packages='randomForest') %dopar% randomForest(x$trainsub,x$classtr,ntree=ntree)
			RFsub <- combine(RFsub, RFsubmid)
			cur_OOBerror=1-sum(predict(RFsub,type='response')==x$classtr)/nrow(x$trainsub)
			iter=iter+1
		}
 */
        int iteration=1;
        int nofTreeStep=50;
        double curOOBerror=0;
        double prevOOBerror=1;
//Build first model
        TunedRandomForest rf= new TunedRandomForest();
        rf.setNumTrees(nofTreeStep);
        rf.buildClassifier(data);
        curOOBerror=rf.measureOutOfBagError();
        while(iteration<20 && curOOBerror< (1-TOLERANCE)*prevOOBerror){
//Add in nofTreeStep models
            rf.addTrees(nofTreeStep,data);
//Find new OOB error. This is probably not updated?             
            double a=rf.measureOutOfBagError();
            prevOOBerror=curOOBerror;
            curOOBerror=a;
        }
        return rf;
    }

    private void cloneToThis(TSBF other){
        numBins=other.numBins;      //bin size for codebook generation   
        oobError=other.oobError;
        z=other.z;
//Variables, dont need to be global, can be local to buildClassifier
        folds=other.folds;
        seriesLength=other.seriesLength;   //data specific       
        numIntervals=other.numIntervals;   //nos intervals per sub series=(int)((zLevel*seriesLength)/minIntervalLength);  
        numSubSeries=other.numSubSeries;         //nos subseries =  (int)(seriesLength/minIntervalLength)-numIntervals;
        minSubLength=other.minSubLength;   // min series length = zlevel*seriesLength
        numOfTreeStep=other.numOfTreeStep; //step size for tree building process
        paramSearch=other.paramSearch;
        trainAcc=other.trainAcc;
        stepWise=other.stepWise;
        subSeries=other.subSeries;
        intervals=other.intervals;
        subseriesRandomForest=other.subseriesRandomForest;
        finalRandForest=other.finalRandForest;
        first=other.first;        
    }
    @Override
    public void buildClassifier(Instances data) throws Exception {
        trainResults.buildTime=System.currentTimeMillis();
        if(numReps>1){
            double bestOOB=1;
            TSBF bestRun=this;
            int r=0;
            for(int i=0;i<numReps;i++){
                TSBF reps=new TSBF();
                reps.numReps=1;
                reps.paramSearch=true;
                reps.buildClassifier(data);
                System.out.println("REP "+i+" ACC = "+reps.trainAcc);
                if(bestOOB>(1-reps.trainAcc)){
                    bestOOB=(1-reps.trainAcc);
                    bestRun=reps;
                    r=i;
                }
                reps=null;
                System.gc();    //Try reduce the memory footprint!
            }
                cloneToThis(bestRun);
                System.out.println("BEST TRAIN ACC="+trainAcc+" REP ="+r);
        }
        else{
            first=new Instances(data,0);
            double bestZ=0;
            double maxAcc=0;
            RandomForest bestFinalModel=null;
            RandomForest bestSubseriesModel=null;
            int[][] bestSubSeries=null;
            int[][][] bestIntervals=null;
            seriesLength=data.numAttributes()-1;
            double [] paras;
            if(paramSearch)
                paras=zLevels;
            else{
                paras=new double[1];
                paras[0]=z;
            }
            for(double zLevel:paras){
//                System.out.println(" ZLEVEL ="+zLevel+" paramSearch ="+paramSearch);
                numIntervals=(int)((zLevel*seriesLength)/minIntervalLength);
                if(numIntervals==0) //Skip this z setting?
                    numIntervals=1;
                minSubLength=minIntervalLength*numIntervals;// Same as  (int)((zLevel*seriesLength)) but clearer
                numSubSeries=  (int)(seriesLength/minIntervalLength)-numIntervals;  //r -d in the paper, very large!
                if(minSubLength<minIntervalLength)  //if minimum subsequence length is smaller than wmin skip this z value 
                    continue;
    //1. Select subsequences and intervals        
                selectSubsequencesAndIntervals();
        //2. Build first transform
                Instances features=formatIntervalInstances(data);
                buildFirstClassificationProblem(data,features);
        //3. Generate class probability estimate for each new instance with a random forest through cross validation
        //  CHANGE THIS TO MATCH PAPER ALGORITHM
    //            subseriesRandomForest=findIncrementalClassifier(features);
         /*       int iteration=1;
                double currentOOBError;
                double prevOOBError=1;
                double TOLERANCE=0.05; 
                                while(iter<20&&cur_OOBerror<(1-tolerance)*prev_OOBerror){    
        */
                double[][] probs;
                if(stepWise){
                    subseriesRandomForest = new TunedRandomForest();
                    subseriesRandomForest.buildClassifier(features);
                    double currentOOBError=subseriesRandomForest.measureOutOfBagError();
                    double prevOOBError=1;
                    int iter=1;
                    while(iter<20&&currentOOBError<(1-TOLERANCE)*prevOOBError){    
    //This implementation is faithful to the original
                        prevOOBError=currentOOBError;
                        ((TunedRandomForest)subseriesRandomForest).addTrees(numOfTreeStep, features);
                        currentOOBError=subseriesRandomForest.measureOutOfBagError();
                    } 
                    probs=((TunedRandomForest)subseriesRandomForest).getOBProbabilities();
                }   
                else{
                    subseriesRandomForest=new RandomForest();
                    subseriesRandomForest.setNumTrees(500);
                    probs=ClassifierTools.crossValidate(subseriesRandomForest,features,folds); 
                    subseriesRandomForest.buildClassifier(features);
                }


        //4. Discretise probabilities into equal width bins, form counts for each instance
        //then concatinate class probabilies to form new set of instances
                int[][][] counts = new int[data.numInstances()][data.numClasses()-1][numBins];
                double[][] classProbs = new double[data.numInstances()][data.numClasses()];
                countsFormat(counts,classProbs,probs,data.numClasses(),data.numInstances());         
                Instances second= formatFrequencyBinInstances(counts,classProbs,data);

    //5. Train a final classifier (random forest). Paper results generated with rand forest 

                double acc=0;    
                if(stepWise){
                     finalRandForest = new TunedRandomForest();
                     finalRandForest.buildClassifier(second);
                     double currentOOBError=finalRandForest.measureOutOfBagError();
                     double prevOOBError=1;
                     int iter=1;
                     while(iter<20&&currentOOBError<(1-TOLERANCE)*prevOOBError){    //The way he has coded it will add in too many trees!
                         prevOOBError=currentOOBError;
                         ((TunedRandomForest)finalRandForest).addTrees(numOfTreeStep, second);
                         currentOOBError=finalRandForest.measureOutOfBagError();
                     } 
                     acc=1-currentOOBError;
                }   
                else{
                    finalRandForest=new RandomForest();    
                    finalRandForest.setNumTrees(500);
                //6. Form a CV estimate of accuracy to choose z value 
                    Random r= new Random();
                    acc=ClassifierTools.stratifiedCrossValidation(data, finalRandForest, 10,r.nextInt());
                }
                if(acc>maxAcc){
                   if(!stepWise)
                        finalRandForest.buildClassifier(second);
                   bestSubseriesModel=subseriesRandomForest;
                   bestFinalModel= finalRandForest;
                   maxAcc=acc;
                   bestZ=zLevel;
                   bestIntervals=intervals;
                   bestSubSeries=subSeries;

                }
            }
    //Reset to the best model
//            System.out.println("Best acc="+maxAcc+" for level "+bestZ+" has "+finalRandForest.getNumTrees()+" trees");
            numIntervals=(int)((bestZ*seriesLength)/minIntervalLength);
            if(numIntervals==0)
                numIntervals=1;
            minSubLength=minIntervalLength*numIntervals;// Same as  (int)((zLevel*seriesLength)) but clearer
            numSubSeries=  (int)(seriesLength/minIntervalLength)-numIntervals;  //r -d in the paper, very large!
            intervals=bestIntervals;
            subSeries=bestSubSeries;
            subseriesRandomForest=bestSubseriesModel;
            finalRandForest=bestFinalModel;
            trainAcc=maxAcc;        
        }
    }
    public void countsFormat(int[][][] counts,double[][] classProbs,double[][] probs,int numClasses, int numInstances){
        for(int i=0;i<numInstances;i++){
            for(int j=0;j<numSubSeries;j++){
                for(int k=0;k<numClasses-1;k++){
//Will need to check for special case 1.0 prob
                    int bin;
                    if(probs[i*numSubSeries+j][k]==1)
                        bin=numBins-1;
                    else
                        bin=(int)(numBins*probs[i*numSubSeries+j][k]);
                    counts[i][k][bin]++;                    
                }
            }
        }
//The relative frequencies of the predicted classes over each series are also concatenated in the codebook        
        for(int i=0;i<numInstances;i++){
            for(int j=0;j<numSubSeries;j++){
                int predicted=0;
                for(int k=1;k<numClasses;k++){
                    if(probs[i*numSubSeries+j][predicted]<probs[i*numSubSeries+j][k])
                        predicted=k;
                }
                classProbs[i][predicted]++;
            }
            for(int k=0;k<numClasses;k++)
               classProbs[i][k]/=numSubSeries; 
        }
        trainResults.buildTime=System.currentTimeMillis()-trainResults.buildTime;
    }
    @Override
    public double[] distributionForInstance(Instance ins) throws Exception{
//Buid first transform
        first.add(ins);
        Instances features=formatIntervalInstances(first);
        buildFirstClassificationProblem(first,features);
//Classify subsequences
        double[][] probs=new double[features.numInstances()][];
        for (int i = 0; i < probs.length; i++) {
            probs[i]=subseriesRandomForest.distributionForInstance(features.instance(i));
        }
//Get histograms
        int[][][] counts = new int[1][ins.numClasses()-1][numBins];
        double[][] classProbs = new double[1][ins.numClasses()];
        countsFormat(counts,classProbs,probs,ins.numClasses(),1);
         
       Instances second= formatFrequencyBinInstances(counts,classProbs,first);
//Construct new instance
        first.remove(0);
//Classify that instance
        return finalRandForest.distributionForInstance(second.get(0));
    }
    private void selectSubsequencesAndIntervals(){
 
//        System.out.println("total series length ="+seriesLength+" min subseries length ="+minIntervalLength+"  z value = "+zLevel+" num subs ="+numSubSeries+" num intervals ="+numIntervals);
        subSeries =new int[numSubSeries][2];
        intervals =new int[numSubSeries][numIntervals][2];
//Find series and intervals      ran  
        for(int i=0;i<numSubSeries;i++){
//Generate subsequences of min length wmin. These are the same for all series
            subSeries[i][0]=rand.nextInt(seriesLength-minSubLength);           
            subSeries[i][1]=rand.nextInt(seriesLength-subSeries[i][0]-minSubLength)+subSeries[i][0]+minSubLength;
            int subSeriesLength=subSeries[i][1]-subSeries[i][0]+1;
 //           System.out.println(" SUBSERIES :"+i+"=["+subSeries[i][0]+","+subSeries[i][1]+"]"+" length ="+subSeriesLength);
//Generate the interval length for the current series.
//				st_sub=floor(runif(0,*lenseries-*minsublength));
//				max_intlen=((*lenseries)-st_sub)/(*nofint);
//				cur_intlen=floor(runif(min_intlen,max_intlen));
            int maxIntLength=subSeriesLength/numIntervals;
            if(maxIntLength<minIntervalLength){
                System.out.println("MAX INT LENGTH < minIntervalLength subseries length ="+subSeriesLength+" num intervals ="+numIntervals+" max int length="+maxIntLength);
                System.exit(0);
            }
//            int minIntLength=minIntervalLength;
//            System.out.println("Max int length="+maxIntLength+" Min int length="+minIntLength);
            int currentIntLength=minIntervalLength;
            if(maxIntLength>minIntervalLength)
                currentIntLength=rand.nextInt(maxIntLength-minIntervalLength+1)+minIntervalLength;
//            System.out.println(" current length ="+currentIntLength);
//Generate intervals. The length of the intervals is randomised, but then that should effect the number??                
 // Seems to ignore the end bit not divisible by length                        
// What happens if this exceed the length of the subseries? 
            for(int j=0;j<numIntervals;j++){
                intervals[i][j][0]=subSeries[i][0]+j*currentIntLength;
                intervals[i][j][1]=subSeries[i][0]+(j+1)*currentIntLength-1;
                if(intervals[i][j][1]>subSeries[i][1]){
                    System.out.println("\t INTERVAL "+j+"["+intervals[i][j][0]+","+intervals[i][j][1]+"] EXCEEDS SUBSERIES "+subSeries[i][0]+","+subSeries[i][1]+"]");
                    System.out.println("\t\t Max interval length ="+maxIntLength+" min interval length ="+minIntervalLength); 
                }
            }
        }
    }

    private void buildFirstClassificationProblem(Instances data, Instances features){
        int instPos=0;
//        System.out.println(" Number of subsequences ="+numSubSeries+"number of intervals per subsequence ="+numIntervals+" number of cases ="+data.numInstances()+" new number of cases ="+features.numInstances());
        for(int k=0;k<data.numInstances();k++){// Instance ins:data){
            double[] series=data.instance(k).toDoubleArray();
 //           if(k==0)
 //               System.out.println("INSTANCE 0="+data.instance(0));
  //          System.out.println(" Series length ="+(series.length-1));
            for(int i=0;i<numSubSeries;i++){
                int pos=0;
//                if(k==0)
 //                   System.out.println(" Setting subseries "+i+" ["+subSeries[i][0]+","+subSeries[i][1]+"]");
//Get whole subseries instance subseries features
                Instance newIns=features.get(instPos++);
                FeatureSet f=new FeatureSet();
                f.setFeatures(series,subSeries[i][0], subSeries[i][1]);
 //               if(k==0)
 //              System.out.println("New num features ="+newIns.numAttributes()+" Whole subsequence features ="+f);
                newIns.setValue(pos++,f.mean);
                newIns.setValue(pos++,f.stDev);
                newIns.setValue(pos++,f.slope);
//Add start and end point
                newIns.setValue(pos++,subSeries[i][0]);
                newIns.setValue(pos++,subSeries[i][1]);
                
//Get interval features                
                for(int j=0;j<numIntervals;j++){
 //               if(k==0)
  //                 System.out.println(" Setting interval "+j+" ["+intervals[i][j][0]+","+intervals[i][j][1]+"]");
                f.setFeatures(series, intervals[i][j][0],intervals[i][j][1]);
                newIns.setValue(pos++,f.mean);
                newIns.setValue(pos++,f.stDev);
                newIns.setValue(pos++,f.slope);
                }
               
            }
        }
       if(InstanceTools.hasMissing(features)){
           System.out.println(" MISSING A VALUE");
           for(int i=0;i<features.numInstances();i++){
               if(features.instance(i).hasMissingValue()){
                   System.out.println("Instance ="+features.instance(i)+" from original instance  "+i/numSubSeries+" ::"+data.instance(i/numSubSeries));
                   System.out.println("\tSubsequence = ["+subSeries[i%numSubSeries][0]+","+subSeries[i%numSubSeries][1]+"]");
                   for(int j=0;j<numIntervals;j++){
                        System.out.println("\t\t interval "+j+" ["+intervals[i%numSubSeries][j][0]+","+intervals[i%numSubSeries][j][1]+"]");
                   }
               }
           }
//       System.out.println(" new data ="+features);
           System.exit(0);
       }
        
    }
    public static void testBinMaker(){
      double[][] probs={{0.05,0.83,0.12},{0.25,0.73,0.02},{0.25,0.13,0.62},{0.1,0.1,0.8},{1,0,0},{0.5,0.2,0.3}};
//4. Discretise probabilities into equal width bins, form counts for each instance
      int numClasses=3;
      int numBins=10;
      int numInstances=2;
      int numSubSeries=3;
        int[][][] counts = new int[numInstances][numClasses-1][numBins];
        for(int i=0;i<numInstances;i++){
            for(int j=0;j<numSubSeries;j++){
                for(int k=0;k<numClasses-1;k++){
//Will need to check for special case 1.0 prob
                    int bin;
                    if(probs[i*numSubSeries+j][k]==1)
                        bin=numBins-1;
                    else
                        bin=(int)(numBins*probs[i*numSubSeries+j][k]);
                    counts[i][k][bin]++;                    
                }
            }
        }
//The relative frequencies of the predicted classes over each series are also concatenated in the codebook        
        double[][] classProbs = new double[numInstances][numClasses];
        for(int i=0;i<numInstances;i++){
            for(int j=0;j<numSubSeries;j++){
//Find predicted class
                int predicted=0;
                for(int k=1;k<numClasses;k++){
                    if(probs[i*numSubSeries+j][predicted]<probs[i*numSubSeries+j][k])
                        predicted=k;
                }
                System.out.println(" instance "+i+" subseries "+j+" predicted ="+predicted);
                classProbs[i][predicted]++;
            }
            for(int k=0;k<numClasses;k++)
               classProbs[i][k]/=numSubSeries; 
        }
        
         for(int i=0;i<numInstances;i++){
             System.out.println("COUNTS INSTANCE "+i);
            for(int k=0;k<numClasses-1;k++){
                System.out.print(" CLASS = "+k+" :::: ");
                for(int j=0;j<numBins;j++){
                    System.out.print(counts[i][k][j]+",");
                }               
            }
               System.out.print(" CLASS PROBS ::");
                for(int j=0;j<numClasses;j++)
                    System.out.print(classProbs[i][j]+",");
                System.out.print("\n");
         }
//        System.out.print(probs);
 //      Instances second= formatFrequencyBinInstances(counts,classProbs);

    }
    public static void main(String[] args) throws Exception {
        String s= "Beef";
        System.out.println(" PROBLEM ="+s);
        Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+s+"\\"+s+"_TRAIN");
        Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+s+"\\"+s+"_TEST");
        TSBF tsbf=new TSBF();
        double a =ClassifierTools.singleTrainTestSplitAccuracy(tsbf, train, test);
        System.out.println(" TEST Acc ="+a);

        
//        DataSets.resultsPath=DataSets.clusterPath+"Results/";
//        DataSets.problemPath=DataSets.clusterPath+"TSC Problems/";        
//        recreatePublishedResults();
  //      testBinMaker();
        System.exit(0);
        DecimalFormat df = new DecimalFormat("##.###");
        try{
            for(int i=1;i<2;i++){
                s="TwoLeadECG";
                System.out.println(" PROBLEM ="+s);
                train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+s+"\\"+s+"_TRAIN");
                test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+s+"\\"+s+"_TEST");
//                RandomForest rf=new RandomForest();
 //               rf.buildClassifier(train);
//                System.out.println(" bag percent ="+rf.getBaggingPercent()+" OOB error "+rf.measureOutOfBagError());
//                System.exit(0);
                tsbf=new TSBF();
                a =ClassifierTools.singleTrainTestSplitAccuracy(tsbf, train, test);
                System.out.println(" error ="+df.format(1-a));
//                tsbf.buildClassifier(train);
 //               double c=tsbf.classifyInstance(test.instance(0));
 //               System.out.println(" Class ="+c);
            }
        }catch(Exception e){
            System.out.println("Exception "+e);
            e.printStackTrace();
            System.exit(0);
        }
    }
}
