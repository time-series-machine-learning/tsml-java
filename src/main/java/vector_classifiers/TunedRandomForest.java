/*
This classifier is enhanced so that classifier builds a random forest with the 
facility to build by forward selection addition of trees to minimize OOB error, 
by far the fastest way.    

As far as tuning are concerned, RandomForest has three parameters

m_MaxDepth: defaults to 0 (no limit on depth)
m_numFeatures: defaults to log(m)+1 (not sqrt(m) as most implementations do
m_numTrees: defaults to 10 


Further enhanced to allow for selection through OOB error estimates and predictions

Further changes: 
1. set number of trees (m_numTrees) via grid search on a range (using OOB) that
defaults to 
{10 [Weka Default],100,200,.., 500 [R default],...,1000} (11 values)
2. set number of features  (max value m==numAttributes without class)
per tree (m_numFeatures) and m_numTrees through grid
search on a range 
1, 10, sqrt(m) [R default], log_2(m)+1 [Weka default], m [full set]}
(4 values)+add an option to choose randomly for each tree?
grid search is then just 55 values and because it uses OOB no CV is required
 */
package vector_classifiers;


import development.CollateResults;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import timeseriesweka.classifiers.ParameterSplittable;
import utilities.ClassifierTools;
import utilities.CrossValidator;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.meta.Bagging;
import utilities.ClassifierResults;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author aj
 */
public class TunedRandomForest extends RandomForest implements SaveParameterInfo, TrainAccuracyEstimate,SaveEachParameter,ParameterSplittable{
    boolean tuneParameters=true;
    int[] paraSpace1;//Maximum tree depth, m_MaxDepth
    int[] paraSpace2;//Number of features per tree,m_numFeatures
    int[] paraSpace3;//Number of trees, m_numTrees
    int[] paras;
    int maxPerPara=10;
    String trainPath="";
    int seed; //need this to seed cver/the forests for consistency in meta-classification/ensembling purposes
    Random rng; //legacy, 'seed' still (and always has) seeds this for any other rng purposes, e.g tie resolution
    ArrayList<Double> accuracy;
    boolean crossValidate=true;
    boolean estimateAcc=true;  //If there is no tuning, this will find the estimate with the fixed values
    private long combinedBuildTime;
    protected String resultsPath;
    protected boolean saveEachParaAcc=false;
//Need to know this before build if the parameters are going to be set by index
//It is only used in the method setParametersFromIndex, which throws an
//Exception if it is zero    
    private int numFeaturesInProblem=0; 
    private static int MAX_FOLDS=10;
    
     private ClassifierResults res =new ClassifierResults();
   public void setNumFeaturesInProblem(int m){
       numFeaturesInProblem=m;
   }
   public void setNumFeaturesForEachTree(int m){
       m_numFeatures=m;
   }
/**
 * Determines whether an estimate of the accuracy is to be obtained from the train data
 * by 10x cross validation
 * @param b 
 */   
    public void setCrossValidate(boolean b){
        if(b)
            setEstimateAcc(b);
        crossValidate=b;
    }
    public void setEstimateAcc(boolean b){
        estimateAcc=b;
    }
    
//methods from SaveEachParameter    
    @Override
    public void setPathToSaveParameters(String r){
            resultsPath=r;
            setSaveEachParaAcc(true);
    }
    @Override
    public void setSaveEachParaAcc(boolean b){
        saveEachParaAcc=b;
    }
//MEthods from ParameterSplittable    
    @Override
    public String getParas() { //This is redundant really.
        return getParameters();
    }
    @Override
    public double getAcc() {
        return res.acc;
    }
    @Override
    public void setParametersFromIndex(int x) {
        tuneParameters=false;
//Three paras, evenly distributed, 1 to maxPerPara.
//Note that if maxPerPara > numFeaturesInProblem, we have a problem, so it will throw an exception later        
        paras=new int[3];
        if(x<1 || x>maxPerPara*maxPerPara*maxPerPara)//Error, invalid range
            throw new UnsupportedOperationException("ERROR parameter index "+x+" out of range for PolyNomialKernel"); //To change body of generated methods, choose Tools | Templates.
        int numLevelsIndex=(x-1)/(maxPerPara*maxPerPara);
        int numFeaturesIndex=((x-1)/(maxPerPara))%maxPerPara;
        int numTreesIndex=x%maxPerPara;
//Need to know number of attributes        
        if(numFeaturesInProblem==0)
            throw new RuntimeException("Error in TunedRandomForest in set ParametersFromIndex: we do not know the number of attributes, need to call setNumFeaturesInProblem before this call");
//Para 1. Maximum tree depth, m_MaxDepth
        if(numLevelsIndex==0)
            paras[0]=0;
        else
            paras[0]=numLevelsIndex*(numFeaturesInProblem/maxPerPara);
//Para 2. Num features
        if(numFeaturesIndex==0)
            paras[1]=(int)Math.sqrt(numFeaturesInProblem);
        else if(numFeaturesIndex==1)
            paras[1]=(int) Utils.log2(numFeaturesInProblem)+1;
        else
            paras[1]=((numFeaturesIndex-1)*numFeaturesInProblem)/maxPerPara;
        if(numTreesIndex==0)
            paras[2]=10; //Weka default
        else
            paras[2]=100*numTreesIndex;
        setMaxDepth(paras[0]);
        setNumFeaturesForEachTree(paras[1]);
        setNumTrees(paras[2]);
        if(m_Debug)
            System.out.println("Index ="+x+" Num Features ="+numFeaturesInProblem+" Max Depth="+paras[0]+" Num Features ="+paras[1]+" Num Trees ="+paras[2]);
    }

   
   @Override
    public ClassifierResults getTrainResults(){
//Temporary : copy stuff into res.acc here
        return res;
    }  
//SaveParameterInfo    
    @Override
    public String getParameters() {
        String result="BuildTime,"+res.buildTime+",CVAcc,"+res.acc+",";
        result+="MaxDepth,"+this.getMaxDepth()+",NumFeatures,"+this.getNumFeatures()+",NumTrees,"+this.getNumTrees();
        return result;
    }

    @Override
    public void setParamSearch(boolean b) {
        tuneParameters=b;
    }
    
    
    public TunedRandomForest(){
        super();
        m_numTrees=500;
        m_numExecutionSlots=1; 
        m_bagger=new EnhancedBagging();
        rng=new Random();
        seed=0;
        accuracy=new ArrayList<>();
    }
    @Override
    public void setSeed(int s){
        super.setSeed(s);
        seed = s;
        rng=new Random();
        rng.setSeed(seed);
    }
    
    public void debug(boolean b){
        m_Debug=b;
    }
    public void tuneParameters(boolean b){
        tuneParameters=b;
    }
    public void setNumTreesRange(int[] d){
        paraSpace1=d;
    }
    public void setNumFeaturesRange(int[] d){
        paraSpace2=d;
    }
 @Override
    public void writeCVTrainToFile(String train) {
        trainPath=train;
        estimateAcc=true;
    }    
 @Override
    public void setFindTrainAccuracyEstimate(boolean setCV){
        estimateAcc=setCV;
    }
    
    
    @Override
    public boolean findsTrainAccuracyEstimate(){ return estimateAcc;}
    protected final void setStandardParaSearchSpace(int m){
//Need to know the number  of features to do this
//Does 1000 parameter searches on a 10x10x10 grid    
       if(m<maxPerPara)
           maxPerPara=m;
        if(m_Debug){
            System.out.println("Number of features ="+m+" max para values ="+maxPerPara);
            System.out.println("Setting defaults ....");
        }
//Para 1. Maximum tree depth, m_MaxDepth
        paraSpace1=new int[maxPerPara];
        paraSpace1[0]=0; // No limit
        for(int i=1;i<paraSpace1.length;i++)
            paraSpace1[i]=paraSpace1[i-1]+m/(paraSpace1.length-1);
//Para 2. Num features
        paraSpace2=new int[maxPerPara];
        paraSpace2[0]=(int)Math.sqrt(m);
        paraSpace2[1]=(int) Utils.log2(m)+1;
        
        for(int i=2;i<maxPerPara;i++)
            paraSpace2[i]=((i-1)*m)/maxPerPara;
 //Para 3. Num trees       
        paraSpace3=new int[10];//Num trees
        paraSpace3[0]=10; //Weka default
        for(int i=1;i<paraSpace3.length;i++)
            paraSpace3[i]=100*i;
        if(m_Debug){
            System.out.print(" m ="+m);
            System.out.print("Para 1 (Num levels) : ");
            for(int i:paraSpace1)
                System.out.print(i+", ");
            System.out.print("\nPara 2 (Num features) : ");
            for(int i:paraSpace2)
                System.out.print(i+", ");
            System.out.print("\nPara 3 (Num trees) : ");
            for(int i:paraSpace3)
                System.out.print(i+", ");
        }
  }
    public void tuneRandomForest(Instances train) throws Exception {
         paras=new int[3];
        int folds=MAX_FOLDS;
        if(folds>train.numInstances())
            folds=train.numInstances();
        double minErr=1;
        this.setSeed(rng.nextInt());
        Instances trainCopy=new Instances(train);
        CrossValidator cv = new CrossValidator();
        cv.setSeed(seed);
        cv.setNumFolds(folds);
        cv.buildFolds(trainCopy);
        ArrayList<TunedSVM.ResultsHolder> ties=new ArrayList<>();
        ClassifierResults tempResults;
        int count=0;
        OutFile temp=null;
        for(int p1:paraSpace1){//Maximum tree depth, m_MaxDepth
            for(int p2:paraSpace2){//Num features
                for(int p3:paraSpace3){//Num trees
                    count++;
                    if(saveEachParaAcc){// check if para value already done
                        File f=new File(resultsPath+count+".csv");
                        if(f.exists()){
                            if(CollateResults.validateSingleFoldFile(resultsPath+count+".csv")==false){
                                System.out.println("Deleting file "+resultsPath+count+".csv because size ="+f.length());
                            }
                            else
                                continue;//If done, ignore skip this iteration                        
                        }
                    }
                    TunedRandomForest model = new TunedRandomForest();
                    model.setMaxDepth(p1);
                    model.setNumFeatures(p2);
                    model.setNumTrees(p3);
                    model.tuneParameters=false;
                    model.estimateAcc=false;
                    model.setSeed(count);
                    tempResults=cv.crossValidateWithStats(model,trainCopy);
                    tempResults.setName("RandFPara"+count);
                    tempResults.setParas("maxDepth,"+p1+",numFeatures,"+p2+",numTrees,"+p3);
                    
                    double e=1-tempResults.acc;
                    if(m_Debug)
                        System.out.println("Depth="+p1+",Features"+p2+",Trees="+p3+" Acc = "+(1-e));
                    accuracy.add(tempResults.acc);
                    if(saveEachParaAcc){// Save to file and close
                        temp=new OutFile(resultsPath+count+".csv");
                        temp.writeLine(tempResults.writeResultsFileToString());
                        temp.closeFile();
                        File f=new File(resultsPath+count+".csv");
                        if(f.exists())
                            f.setWritable(true, false);
                        
                    }                
                    else{
                        if(e<minErr){
                        minErr=e;
                        ties=new ArrayList<>();//Remove previous ties
                        ties.add(new TunedSVM.ResultsHolder(p1,p2,p3,tempResults));
                        }
                        else if(e==minErr){//Sort out ties
                            ties.add(new TunedSVM.ResultsHolder(p1,p2,p3,tempResults));
                        }
                    }
                }
            }
        }
        int bestNumLevels;
        int bestNumFeatures;
        int bestNumTrees;
        
        minErr=1;
        if(saveEachParaAcc){
// Check they are all there first. 
            int missing=0;
            for(int p1:paraSpace1){
                for(int p2:paraSpace2){
                    for(int p3:paraSpace3){
                        File f=new File(resultsPath+count+".csv");
                        if(!(f.exists() && f.length()>0))
                            missing++;
                    }
                }
              }
            
            if(missing==0)//All present
            {
                combinedBuildTime=0;
    //            If so, read them all from file, pick the best
                count=0;
                for(int p1:paraSpace1){//C
                    for(int p2:paraSpace2){//Exponent
                        for(int p3:paraSpace3){//B
                            count++;
                            tempResults = new ClassifierResults();
                            tempResults.loadFromFile(resultsPath+count+".csv");
                            combinedBuildTime+=tempResults.buildTime;
                            double e=1-tempResults.acc;
                            if(e<minErr){
                                minErr=e;
                                ties=new ArrayList<>();//Remove previous ties
                                ties.add(new TunedSVM.ResultsHolder(p1,p2,p3,tempResults));
                            }
                            else if(e==minErr){//Sort out ties
                                    ties.add(new TunedSVM.ResultsHolder(p1,p2,p3,tempResults));
                            }
        //Delete the files here to clean up.

                            File f= new File(resultsPath+count+".csv");
                            if(!f.delete())
                                System.out.println("DELETE FAILED "+resultsPath+count+".csv");
                        }
                    }            
                }
                TunedSVM.ResultsHolder best=ties.get(rng.nextInt(ties.size()));
                bestNumLevels=(int)best.x;
                bestNumFeatures=(int)best.y;
                bestNumTrees=(int)best.z;
                paras[0]=bestNumLevels;
                paras[1]=bestNumFeatures;
                paras[2]=bestNumTrees;
                this.setMaxDepth(bestNumLevels);
                this.setNumFeatures(bestNumFeatures);
                this.setNumTrees(bestNumTrees);
               
                res=best.res;
                if(m_Debug)
                    System.out.println("Bestnum levels ="+bestNumLevels+" best num features = "+bestNumFeatures+" best num trees ="+bestNumTrees+" best train acc = "+res.acc);
            }else//Not all present, just ditch
                System.out.println(resultsPath+" error: missing  ="+missing+" parameter values");
        }
        else{
            TunedSVM.ResultsHolder best=ties.get(rng.nextInt(ties.size()));
            bestNumLevels=(int)best.x;
            bestNumFeatures=(int)best.y;
            bestNumTrees=(int)best.z;
            paras[0]=bestNumLevels;
            paras[1]=bestNumFeatures;
            paras[2]=bestNumTrees;
            this.setMaxDepth(bestNumLevels);
            this.setNumFeatures(bestNumFeatures);
            this.setNumTrees(bestNumTrees);
            res=best.res;
         }     
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception{
        long startTime=System.currentTimeMillis(); 
//********* 1: Set up the main classifier with standard Weka calls ***************/      
    // can classifier handle the data?
        getCapabilities().testWithFail(data);
        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
//this is only used if CV is used to find parameters or estimate acc from train data        
        int folds=10;
        if(folds>data.numInstances())
            folds=data.numInstances();
        super.setSeed(seed);
        super.setNumFeatures((int)Math.sqrt(data.numAttributes()-1));
/******* 2. Tune parameters if required: 
 * 
 * NOTE: the number of trees could be found incrementally, just start with the smallest
 * number and add in each time rather than rebuild. It would massively speed up the search
 * this has been implemented for the EnhancedBagger,but is not yet used. 
 * Obviously cannot do this for the number of attributes
 */
        if(tuneParameters){
            if(paraSpace1==null)
                setStandardParaSearchSpace(data.numAttributes()-1);
            tuneRandomForest(data);
        }
        else //Override WEKA's default which is worse than sqrt(m)
            setNumFeatures(Math.max(1,(int)Math.sqrt(data.numAttributes()-1)));
        
/******** 3. Build final classifier ***************/        
/*Cant call super.buildClassifier as it resets the bagger to Bagging instead of 
 EnhancedBagging so instead straight cut and paste from RandomForest, with 
 Bagging changed to EnhancedBagging and default size changed
*/
        m_bagger = new EnhancedBagging();
//Test with C4.5 rather than RT        
        RandomTree rTree = new RandomTree();
//        J48 rTree = new J48();

        // set up the random tree options
        
        
        if(m_numFeatures>data.numAttributes()-1)
            m_numFeatures=data.numAttributes()-1;
        if(m_MaxDepth>data.numAttributes()-1)
            m_MaxDepth=0;
        m_KValue = m_numFeatures;
//the value in m_numFeatures is not actually used 
//its only role is setting m_KValue
        rTree.setKValue(m_KValue);
        rTree.setMaxDepth(getMaxDepth());
        // set up the bagger and build the forest
        m_bagger.setClassifier(rTree);
        m_bagger.setSeed(seed);
        m_bagger.setNumIterations(m_numTrees);
        m_bagger.setCalcOutOfBag(true);
        m_bagger.setNumExecutionSlots(m_numExecutionSlots);
        m_bagger.buildClassifier(data);       
        
/*** 4. Find the estimates of the train acc, either through CV or OOB  ****/
// do this after the main build in case OOB is used, because we need the main 
//classifier for that
//NOTE IF THE CLASSIFIER IS TUNED THIS WILL BE A POSSIBLE SOURCE OF BIAS
//It should really be nested up a level.         
 
    
        if(estimateAcc){   //Need find train acc, either through CV or OOB
            if(crossValidate){  
                RandomForest t= new RandomForest();
                t.setNumFeatures(this.getNumFeatures());
                t.setNumTrees(this.getNumTrees());
                t.setSeed(seed);
                CrossValidator cv = new CrossValidator();
                cv.setSeed(seed); 
                cv.setNumFolds(folds);
                cv.buildFolds(data);
                res = cv.crossValidateWithStats(t, data);
                if(m_Debug){
                    System.out.println("In cross  validate");
                    System.out.println(getParameters());
                }
            }
            else{
                res.acc=1-this.measureOutOfBagError();
//Get OOB probabilities. This is not possible with the standard 
//random forest bagger, hence the use of EnhancedBagger
                System.out.println("BAGGER CLASS = "+m_bagger.getClass().getName());
                
                ((EnhancedBagging)m_bagger).findOOBProbabilities();
                double[][] OOBPredictions=((EnhancedBagging)m_bagger).OOBProbabilities;
                for(int i=0;i<data.numInstances();i++)
                    res.storeSingleResult(data.instance(i).classValue(),OOBPredictions[i]);
            }
        }
        
        res.buildTime=System.currentTimeMillis()-startTime;
        if(trainPath!=""){  //Save basic train results
            OutFile f= new OutFile(trainPath);
            f.writeLine(data.relationName()+",TunedRandF,Train");
            f.writeLine(getParameters());
            f.writeLine(res.acc+"");
            f.writeLine(res.writeInstancePredictions());
        } 
        
    }

    public void addTrees(int n, Instances data) throws Exception{
        EnhancedBagging newTrees =new EnhancedBagging();
        RandomTree rTree = new RandomTree();
        // set up the random tree options
        m_KValue = m_numFeatures;
        rTree.setKValue(m_KValue);
        rTree.setMaxDepth(getMaxDepth());
//Change this so that it is reproducable
        Random r= new Random();
        newTrees.setSeed(r.nextInt());
        newTrees.setClassifier(rTree);
        newTrees.setNumIterations(n);
        newTrees.setCalcOutOfBag(true);
        newTrees.setNumExecutionSlots(m_numExecutionSlots);
        newTrees.buildClassifier(data);
        newTrees.findOOBProbabilities();
//Merge with previous
        m_bagger.aggregate(newTrees);
        m_bagger.finalizeAggregation();
//Update OOB Error, as this is seemingly not done in the bagger
        
        m_numTrees+=n;
        m_bagger.setNumIterations(m_numTrees); 
        ((EnhancedBagging)m_bagger).mergeBaggers(newTrees);
        
    }
    public double getBaggingPercent(){
      return m_bagger.getBagSizePercent();
    }

    static protected class EnhancedBagging extends Bagging{
// 
        @Override
        public void buildClassifier(Instances data)throws Exception {
            super.buildClassifier(data);
            m_data=data;
//            System.out.println(" RESET BAGGER");

        }
        double[][] OOBProbabilities;
        int[] counts;
        public void mergeBaggers(EnhancedBagging other){
            for (int i = 0; i < m_data.numInstances(); i++) {
                for (int j = 0; j < m_data.numClasses(); j++) {
                      OOBProbabilities[i][j]=counts[i]*OOBProbabilities[i][j]+other.counts[i]*other.OOBProbabilities[i][j];
                      OOBProbabilities[i][j]/=counts[i]+other.counts[i];
                }
                counts[i]=counts[i]+other.counts[i];
            }
//Merge  m_inBags index i is classifier, j the instance
            boolean[][] inBags = new boolean[m_inBag.length+other.m_inBag.length][];
            for(int i=0;i<m_inBag.length;i++)
                inBags[i]=m_inBag[i];
            for(int i=0;i<other.m_inBag.length;i++)
                inBags[m_inBag.length+i]=other.m_inBag[i];
            m_inBag=inBags;
            findOOBError();
        }
        public void findOOBProbabilities() throws Exception{
            OOBProbabilities=new double[m_data.numInstances()][m_data.numClasses()];
            counts=new int[m_data.numInstances()];
            for (int i = 0; i < m_data.numInstances(); i++) {
                for (int j = 0; j < m_Classifiers.length; j++) {
                    if (m_inBag[j][i])
                      continue;
                    counts[i]++;
                    double[] newProbs = m_Classifiers[j].distributionForInstance(m_data.instance(i));
                // average the probability estimates
                    for (int k = 0; k < m_data.numClasses(); k++) {
                        OOBProbabilities[i][k] += newProbs[k];
                    }
                }
                for (int k = 0; k < m_data.numClasses(); k++) {
                    OOBProbabilities[i][k] /= counts[i];
                }
            }
        }
        
        public double findOOBError(){
            double correct = 0.0;
            for (int i = 0; i < m_data.numInstances(); i++) {
                double[] probs = OOBProbabilities[i];
                int vote =0;
                for (int j = 1; j < probs.length; j++) {
                  if(probs[vote]<probs[j])
                      vote=j;
            }
            if(m_data.instance(i).classValue()==vote) 
                correct++;
            }
            m_OutOfBagError=1- correct/(double)m_data.numInstances();
//            System.out.println(" NEW OOB ERROR ="+m_OutOfBagError);
            return m_OutOfBagError;
        }
        
 //       public double getOOBError
    }
    public double findOOBError() throws Exception{
        ((EnhancedBagging)m_bagger).findOOBProbabilities();
        return ((EnhancedBagging)m_bagger).findOOBError();
    }
    public double[][] findOOBProbabilities() throws Exception{
        ((EnhancedBagging)m_bagger).findOOBProbabilities();
        return ((EnhancedBagging)m_bagger).OOBProbabilities;
    }
    public double[][] getOBProbabilities() throws Exception{
        return ((EnhancedBagging)m_bagger).OOBProbabilities;
    }
  
    public static void jamesltests() {
        //tests to confirm correctness of cv changes
        //summary: pre/post change avg accs over 50 folds
        // train: 0.9689552238805973 vs 0.9680597014925374
        // test: 0.9590670553935859 vs 0.9601943634596699
        
        //post change trainaccs/testaccs on 50 folds of italypowerdemand: 
//                trainacc=0.9680597014925374
//        folds:
//        [0.9552238805970149, 0.9552238805970149, 1.0, 1.0, 0.9552238805970149, 
//        0.9701492537313433, 0.9552238805970149, 0.9402985074626866, 0.9552238805970149, 1.0, 
//        0.9552238805970149, 0.9701492537313433, 0.9701492537313433, 0.9701492537313433, 0.9850746268656716, 
//        0.9552238805970149, 0.9402985074626866, 0.9850746268656716, 1.0, 0.9552238805970149, 
//        0.9850746268656716, 0.9701492537313433, 0.9701492537313433, 0.9552238805970149, 1.0, 
//        0.9701492537313433, 0.9701492537313433, 0.9552238805970149, 0.9552238805970149, 0.9850746268656716, 
//        0.9402985074626866, 0.9850746268656716, 1.0, 0.9850746268656716, 0.9850746268656716, 
//        0.9402985074626866, 0.9552238805970149, 0.9253731343283582, 0.9701492537313433, 0.9701492537313433, 
//        0.9701492537313433, 1.0, 0.9402985074626866, 0.9701492537313433, 0.9552238805970149, 
//        0.9402985074626866, 0.9701492537313433, 0.9552238805970149, 0.9850746268656716, 0.9701492537313433]
//
//                testacc=0.9601943634596699
//        folds:
//        [0.9543245869776482, 0.9271137026239067, 0.9582118561710399, 0.966958211856171, 0.9718172983479106, 
//        0.9650145772594753, 0.9582118561710399, 0.9689018464528668, 0.9494655004859086, 0.966958211856171, 
//        0.9620991253644315, 0.9698736637512148, 0.9659863945578231, 0.9659863945578231, 0.966958211856171, 
//        0.9718172983479106, 0.9416909620991254, 0.9640427599611273, 0.9368318756073858, 0.9698736637512148,
//        0.966958211856171, 0.9494655004859086, 0.9582118561710399, 0.9698736637512148, 0.9620991253644315, 
//        0.9650145772594753, 0.9640427599611273, 0.9601554907677357, 0.9319727891156463, 0.967930029154519, 
//        0.9523809523809523, 0.967930029154519, 0.9591836734693877, 0.9727891156462585, 0.9572400388726919, 
//        0.9329446064139941, 0.9718172983479106, 0.9620991253644315, 0.9689018464528668, 0.9514091350826045, 
//        0.9630709426627794, 0.966958211856171, 0.9543245869776482, 0.9718172983479106, 0.9698736637512148, 
//        0.9552964042759962, 0.9727891156462585, 0.9329446064139941, 0.9630709426627794, 0.9650145772594753]

        //pre change trainaccs/testaccs on 50 folds of italypowerdemand: 
//                trainacc=0.9689552238805973
//        folds:
//        [0.9402985074626866, 0.9701492537313433, 1.0, 1.0, 0.9253731343283582, 
//        0.9850746268656716, 0.9850746268656716, 0.9552238805970149, 0.9552238805970149, 1.0, 
//        0.9402985074626866, 0.9701492537313433, 0.9701492537313433, 0.9850746268656716, 0.9850746268656716, 
//        0.9701492537313433, 0.9253731343283582, 0.9850746268656716, 1.0, 0.9701492537313433, 
//        0.9850746268656716, 0.9850746268656716, 0.9701492537313433, 0.9701492537313433, 1.0, 
//        0.9552238805970149, 0.9552238805970149, 0.9552238805970149, 0.9701492537313433, 0.9850746268656716, 
//        0.9552238805970149, 0.9850746268656716, 1.0, 0.9850746268656716, 0.9850746268656716, 
//        0.9701492537313433, 0.9552238805970149, 0.9402985074626866, 0.9701492537313433, 0.9552238805970149, 
//        0.9850746268656716, 1.0, 0.9402985074626866, 0.9701492537313433, 0.9402985074626866, 
//        0.9253731343283582, 0.9701492537313433, 0.9552238805970149, 0.9552238805970149, 0.9552238805970149]
//
//                testacc=0.9590670553935859
//        folds:
//        [0.9514091350826045, 0.9290573372206026, 0.9591836734693877, 0.967930029154519, 0.9708454810495627, 
//        0.9689018464528668, 0.9650145772594753, 0.9708454810495627, 0.9358600583090378, 0.967930029154519, 
//        0.9640427599611273, 0.9640427599611273, 0.9630709426627794, 0.9659863945578231, 0.9543245869776482, 
//        0.9689018464528668, 0.9514091350826045, 0.9659863945578231, 0.9659863945578231, 0.9611273080660836,
//        0.9689018464528668, 0.9504373177842566, 0.9504373177842566, 0.9698736637512148, 0.9630709426627794, 
//        0.9620991253644315, 0.9582118561710399, 0.966958211856171, 0.9543245869776482, 0.9640427599611273, 
//        0.9514091350826045, 0.9533527696793003, 0.9659863945578231, 0.9689018464528668, 0.9572400388726919, 
//        0.967930029154519, 0.9689018464528668, 0.9698736637512148, 0.9698736637512148, 0.9582118561710399, 
//        0.9601554907677357, 0.966958211856171, 0.9378036929057337, 0.9689018464528668, 0.9650145772594753, 
//        0.8794946550048591, 0.9737609329446064, 0.9319727891156463, 0.9484936831875608, 0.9689018464528668]


        System.out.println("ranftestsWITHCHANGES");
        
        String dataset = "ItalyPowerDemand";
        
        Instances train = ClassifierTools.loadData("c:/tsc problems/"+dataset+"/"+dataset+"_TRAIN");
        Instances test = ClassifierTools.loadData("c:/tsc problems/"+dataset+"/"+dataset+"_TEST");
        
        int rs = 50;
        
        double[] trainAccs = new double[rs];
        double[] testAccs = new double[rs];
        double trainAcc =0;
        double testAcc =0;
        for (int r = 0; r < rs; r++) {
            Instances[] data = InstanceTools.resampleTrainAndTestInstances(train, test, r);
            
            TunedRandomForest ranF = new TunedRandomForest();
            ranF.setCrossValidate(true);
            ranF.setEstimateAcc(true);
            try {
                ranF.buildClassifier(data[0]);
            } catch (Exception ex) {
                Logger.getLogger(TunedRandomForest.class.getName()).log(Level.SEVERE, null, ex);
            }
            
            trainAccs[r] = ranF.res.acc;
            trainAcc+=trainAccs[r];
            
            testAccs[r] = ClassifierTools.accuracy(data[1], ranF);
            testAcc+=testAccs[r];
            
            System.out.print(".");
        }
        trainAcc/=rs;
        testAcc/=rs;
        
        System.out.println("\nacc="+trainAcc);
        System.out.println(Arrays.toString(trainAccs));
        
        System.out.println("\nacc="+testAcc);
        System.out.println(Arrays.toString(testAccs));
    }
      
    
    public static void main(String[] args) {
        cheatOnMNIST();
        TunedRandomForest randF=new TunedRandomForest();
        randF.m_Debug=true;
        randF.setStandardParaSearchSpace(200);
        
//        randF.setNumFeaturesInProblem(3);
 //       for(int i=1;i<=1000;i++)
 //           randF.setParametersFromIndex(i);
            
  //      jamesltests();
  //      testBinMaker();
       System.exit(0);
        DecimalFormat df = new DecimalFormat("##.###");
        try{
            String dset = "balloons";             
           Instances all=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\UCI Problems\\"+dset+"\\"+dset);        
            Instances[] split=InstanceTools.resampleInstances(all,1,0.5);
                TunedRandomForest rf=new TunedRandomForest();
                rf.debug(true);
                rf.tuneParameters(true);
               rf.buildClassifier(split[0]);
                System.out.println(" bag percent ="+rf.getBaggingPercent()+" OOB error "+rf.measureOutOfBagError());
/*                
                for(int i=0;i<5;i++){
                    System.out.println(" Number f trees ="+rf.getNumTrees()+" num elements ="+rf.numElements());
                    System.out.println(" bag percent ="+rf.getBaggingPercent()+" OOB error "+rf.measureOutOfBagError());
                    double[][] probs=rf.findOOBProbabilities();
/*s
                    for (int j = 0; j < probs.length; j++) {
                        double[] prob = probs[j];
                        for (int k = 0; k < prob.length; k++) {
                            System.out.print(","+prob[k]);
                        }
                        System.out.println("");
                        
                    }

                    rf.addTrees(50, train);
                }
                int correct=0;
                for(Instance ins:test){
                    double[] pred=rf.distributionForInstance(ins);
                    double cls=rf.classifyInstance(ins);
                    if(cls==ins.classValue())
                        correct++;
                }
                System.out.println(" ACC = "+((double)correct)/test.numInstances());
//                System.out.println(" calc out of bag? ="+rf.m_bagger.m_CalcOutOfBag);
                System.exit(0);
                double a =ClassifierTools.singleTrainTestSplitAccuracy(rf, train, test);
                System.out.println(" error ="+df.format(1-a));
//                tsbf.buildClassifier(train);
 //               double c=tsbf.classifyInstance(test.instance(0));
 //               System.out.println(" Class ="+c);
*/
        }catch(Exception e){
            System.out.println("Exception "+e);
            e.printStackTrace();
            System.exit(0);
        }
    }
   
    public static void cheatOnMNIST(){
        Instances train=ClassifierTools.loadData("\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\LargeProblems\\MNIST\\MNIST_TRAIN");
        Instances test=ClassifierTools.loadData("\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\LargeProblems\\MNIST\\MNIST_TEST");
        RandomForest rf=new RandomForest();
        System.out.println("Data loaded ......");
        double a =ClassifierTools.singleTrainTestSplitAccuracy(rf, train, test);
        System.out.println("Trees ="+10+" acc = "+a);
        for(int trees=50;trees<=1000;trees+=50){
            rf.setNumTrees(trees);
            a =ClassifierTools.singleTrainTestSplitAccuracy(rf, train, test);
            System.out.println("Trees ="+trees+" acc = "+a);
        }
        
    }
  
  
}
