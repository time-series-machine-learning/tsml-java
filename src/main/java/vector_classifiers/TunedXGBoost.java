
package vector_classifiers;

import fileIO.OutFile;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import timeseriesweka.classifiers.ParameterSplittable;
import utilities.ClassifierResults;
import utilities.CrossValidator;
import utilities.DebugPrinting;
import utilities.TrainAccuracyEstimate;
import vector_classifiers.SaveEachParameter;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import development.CollateResults;
import development.DataSets;
import development.Experiments;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import weka.classifiers.Classifier;

/**
 * Original code repo, around which this class wraps: https://github.com/dmlc/xgboost
 * Paper: 
        @inproceedings{chen2016xgboost,
         title={Xgboost: A scalable tree boosting system},
         author={Chen, Tianqi and Guestrin, Carlos},
         booktitle={Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining},
         pages={785--794},
         year={2016},
         organization={ACM}
       }
 * 
 * Wrapping around the public xgboost API for multiclass classification, with automatic grid search parameter tuning 
 * as an option. Would search over the learning rate, num iterations, max tree depth, and min child weighting.
 * 
 * TODOS:
 * - Sort out any extra licensing/citations needed
 * - Thorough testing of the tuning checkpointing/para splitting for evaluation
 * - Potentially tweaking the para spaces depending on observed behaviour
 * - Any extra software engineering-type things required
 * - Look for speedups, esp early abandons on grid search with num iters
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class TunedXGBoost extends AbstractClassifier implements SaveParameterInfo,DebugPrinting, TrainAccuracyEstimate, SaveEachParameter, ParameterSplittable {
    int seed = 0;
//    Random rng = null;
    
    //data info
    int numTrainInsts = -1;
    int numAtts = -1;
    int numClasses = -1;
    Instances trainInsts = null;
    DMatrix trainDMat = null;
    
    //model
    HashMap<String, DMatrix> watches = null;
    HashMap<String, Object> params = null;
    Booster booster = null;
    ClassifierResults trainResults =new ClassifierResults();
    
    //hyperparameters - fixed
    float rowSubsampling = 0.8f; //aka rowSubsampling
    float colSubsampling = 0.8f; //aka colsample_bytree
    int minChildWeight = 1; //aka min_child_weight. NO LONGER TUNABLE, LEFT AS DEFAULT ( 1 ), on advice from rotf paper reviewer
    
    //old parameters
//    //hyperparameter settings informed by a mix of these, but also restricted in certain situations
//    //to bring in line with the amount of tuning provided to other classifiers for fairness.
//    //subject to change
//    //      https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
//    //      https://www.slideshare.net/odsc/owen-zhangopen-sourcetoolsanddscompetitions1 (slide 12)
//    //      https://cambridgespark.com/content/tutorials/hyperparameter-tuning-in-xgboost/index.html
//    //hyperparameters - tunable through cv (6*5*5*7 = 1050 possible paras)
//    float learningRate = 0.1f; //aka eta
//    static float[] learningRateParaRange = { 0.001f, 0.01f, 0.05f, 0.1f, 0.2f, 0.3f };
//    int maxTreeDepth = 4; //aka max_depth
//    static int[] maxTreeDepthParaRange = { 2,4,6,8,10 };
//    int minChildWeight = 1; //aka min_child_weight
//    static int[] minChildWeightParaRange = { 1,3,5,7,9 };
//    int numIterations = 500; //aka rounds
//    static int[] numIterationsParaRange = { 50, 100, 250, 500, 1000, 1500, 2000 };
    
    //new parameters, on advice from rotf paper reviewer
    float learningRate = 0.1f; //aka eta
//    static float[] learningRateParaRange = { 0.01f, 0.1f, 0.2f };
    static float[] learningRateParaRange = { 0.00001f, 0.0001f, 0.001f, 0.01f, 0.05f, 0.1f, 0.15f, 0.2f, 0.25f, 0.3f };
    int maxTreeDepth = 4; //aka max_depth
//    static int[] maxTreeDepthParaRange = { 1,3,5 };
    static int[] maxTreeDepthParaRange = { 1,2,3,4,5,6,7,8,9,10 };
    int numIterations = 500; //aka rounds
//    static int[] numIterationsParaRange = { 10, 25, 50};
    static int[] numIterationsParaRange = { 10, 25, 50, 100, 250, 500, 750, 1000, 1250, 1500 };
    
    //tuning/cv/jobsplitting
    int cvFolds = 10;
    boolean tuneParameters=false;
    boolean estimateAcc=true;  //If there is no tuning, this will find the estimate with the fixed values
    protected String resultsPath;
    protected boolean saveEachParaAcc=false;
    ArrayList<Double> paramAccuracies;
    private long combinedBuildTime;
    String trainPath="";
    protected boolean findTrainAcc=true;
    boolean runSingleThreaded = false;
    
    public TunedXGBoost() {
        
    }
    
    public static void setDefaultParaSearchSpace_1000paras() { 
        learningRateParaRange = new float[] { 0.00001f, 0.0001f, 0.001f, 0.01f, 0.05f, 0.1f, 0.15f, 0.2f, 0.25f, 0.3f };
        maxTreeDepthParaRange = new int[] { 1,2,3,4,5,6,7,8,9,10 };
        numIterationsParaRange = new int[]  { 10, 25, 50, 100, 250, 500, 750, 1000, 1250, 1500 };
    }
    
    public static void setSmallParaSearchSpace_64paras() { 
        learningRateParaRange = new float[] { 0.001f, 0.01f, 0.1f, 0.2f };
        maxTreeDepthParaRange = new int[]  { 1,3,5,7 };
        numIterationsParaRange = new int[]  { 250, 500, 1000, 1500};
    }
    
    public int getSeed() { 
        return seed;  
    }
    public void setSeed(int rngSeed) { 
        this.seed = rngSeed;
    }

    public boolean getTuneParameters() {
        return tuneParameters;
    }
    public void setTuneParameters(boolean tuneParameters) {
        this.tuneParameters = tuneParameters;
    }

    public float getLearningRate() {
        return learningRate;
    }
    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public int getMaxTreeDepth() {
        return maxTreeDepth;
    }
    public void setMaxTreeDepth(int maxTreeDepth) {
        this.maxTreeDepth = maxTreeDepth;
    }

    public int getMinChildWeight() {
        return minChildWeight;
    }
    public void setMinChildWeight(int minChildWeight) {
        this.minChildWeight = minChildWeight;
    }

    public int getNumIterations() {
        return numIterations;
    }
    public void setNumIterations(int numIterations) {
        this.numIterations = numIterations;
    }
    
    public boolean getRunSingleThreaded() {
        return runSingleThreaded;
    }
    
    public void setRunSingleThreaded(boolean runSingleThreaded) { 
        this.runSingleThreaded = runSingleThreaded;
    }

    
    //copied over/refactored from tunedsvm/randf/rotf
    public static class XGBoostParamResultsHolder implements Comparable<XGBoostParamResultsHolder> {
        float learningRate;
        int maxTreeDepth;
        int numIterations;
        int conservedness;
        ClassifierResults results;
        
        XGBoostParamResultsHolder(float learningRate, int maxTreeDepth, int numIterations,ClassifierResults r){
            this.learningRate=learningRate;
            this.maxTreeDepth=maxTreeDepth;
            this.numIterations=numIterations;
            
            conservedness = computeConservedness();
            results=r;
        }
        
        @Override
        public String toString() {
            return "learningRate="+learningRate+",maxTreeDepth="+maxTreeDepth+",numIterations="+numIterations+",conservedness="+conservedness+",acc="+results.acc;
        }
        
        /**
         * This values wants to be minimised, higher values = potentially more prone to overfitting
         */
        public int computeConservedness() {
            return (1 + Arrays.binarySearch(TunedXGBoost.learningRateParaRange, learningRate))
                * (1 + Arrays.binarySearch(TunedXGBoost.maxTreeDepthParaRange, maxTreeDepth))
                * (1 + Arrays.binarySearch(TunedXGBoost.numIterationsParaRange, numIterations));
        }
        
        /**
         * Implements a fairly naive way of determining if this param set is more conservative than the other,
         * based on the total 'ranking' of each of the param values within the 4 param spaces. 
         * 
         * Returns less than zero if this is LESS conservative than other (i.e this.computeConservedness() > other.computeConservedness())
         * Returns greater than zero if this is MORE conservative than other (i.e this.computeConservedness() < other.computeConservedness())
         * 
         * Therefore to find most conservative in list of params, use max();
         */
        @Override
        public int compareTo(XGBoostParamResultsHolder other) {
            return other.conservedness - this.conservedness;
        }
    }
    
    //copied over/refactored from vector_classifiers.tunedsvm/randf/rotf
    public void tuneHyperparameters() throws Exception {
        printlnDebug("tuneHyperparameters()");
        
        double minErr=1;
        paramAccuracies=new ArrayList<>();
        
        Instances trainCopy=new Instances(trainInsts);
        CrossValidator cv = new CrossValidator();
        cv.setSeed(seed);
        cv.setNumFolds(cvFolds);
        cv.buildFolds(trainCopy);
        ArrayList<XGBoostParamResultsHolder> ties=new ArrayList<>();
        ClassifierResults tempResults;
        int count=0;
        OutFile temp=null;
        for(float p1:learningRateParaRange){
            for(int p2:maxTreeDepthParaRange){
//                TuningXGBoostCrossValidationWrapper cvmodels = new TuningXGBoostCrossValidationWrapper(p1, p2);
//                cvmodels.setSeed(seed);

                for(int p4:numIterationsParaRange){
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
                    TunedXGBoost model = new TunedXGBoost();
                    model.setLearningRate(p1);
                    model.setMaxTreeDepth(p2);
                    model.setMinChildWeight(minChildWeight);
                    model.setNumIterations(p4);
                    model.tuneParameters=false;
                    model.estimateAcc=false;
                    model.setSeed(seed);
                    tempResults=cv.crossValidateWithStats(model,trainCopy);

//                    cvmodels.setNextNumIterations(p4);
//                    tempResults=cv.crossValidateWithStats(cvmodels,trainCopy);

                    tempResults.setName("XGBoostPara"+count);
                    tempResults.setParas("learningRate,"+p1+",maxTreeDepth,"+p2+",numIterations="+p4);

                    double e=1-tempResults.acc;
                    printlnDebug("learningRate="+p1+",maxTreeDepth"+p2+",numIterations="+p4+" Acc = "+(1-e));
                    paramAccuracies.add(tempResults.acc);
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
                            ties.add(new XGBoostParamResultsHolder(p1,p2,p4,tempResults));
                        }
                        else if(e==minErr)//Sort out ties
                            ties.add(new XGBoostParamResultsHolder(p1,p2,p4,tempResults));
                    }
                }
            }
        }
        
        minErr=1;
        if(saveEachParaAcc){
// Check they are all there first. 
            int missing=0;
            count=1;
            for(float p1:learningRateParaRange){
                for(int p2:maxTreeDepthParaRange){
                    for(int p4:numIterationsParaRange){
                        File f=new File(resultsPath+count+".csv");
                        if(!(f.exists() && f.length()>0))
                            missing++;
                        count++;
                    }
                }
            }
            
            if(missing==0)//All present
            {
                //rebuild the accuracies list
                //if we had checkpointing on, but managed to perform the tuning in a single execution,
                //this will be a waste of time (but functinoally makes no difference), however if we had 
                //to rerun multiple times or got here via para splitting, the list will be empty/incomplete,
                //so start from scratch and repopulate it
                paramAccuracies=new ArrayList<>();
                
                combinedBuildTime=0;
    //            If so, read them all from file, pick the best
                count=0;
                for(float p1:learningRateParaRange){
                    for(int p2:maxTreeDepthParaRange){
                        for(int p4:numIterationsParaRange){
                            count++;
                            
                            tempResults = new ClassifierResults();
                            tempResults.loadFromFile(resultsPath+count+".csv");
                            
                            combinedBuildTime+=tempResults.buildTime;
                            paramAccuracies.add(tempResults.acc);
                            
                            double e=1-tempResults.acc;
                            if(e<minErr){
                                minErr=e;
                                ties=new ArrayList<>();//Remove previous ties
                                ties.add(new XGBoostParamResultsHolder(p1,p2,p4,tempResults));
                            }
                            else if(e==minErr){//Sort out ties
                                ties.add(new XGBoostParamResultsHolder(p1,p2,p4,tempResults));
                            }
        //Delete the files here to clean up.

                            File f= new File(resultsPath+count+".csv");
                            if(!f.delete())
                                System.out.println("DELETE FAILED "+resultsPath+count+".csv");
                        }
                    }            
                }
//                XGBoostParamResultsHolder best=ties.get(rng.nextInt(ties.size()));
                XGBoostParamResultsHolder best=Collections.max(ties); //get the most conservative (see XGBoostParamResultsHolder.computeconservedness())
                printlnDebug("Best learning rate ="+best.learningRate+" best max depth = "+best.maxTreeDepth+" best num iterations ="+best.numIterations+ " acc = " + trainResults.acc + " (num ties = " + ties.size() + ")");
                
                this.setLearningRate(best.learningRate);
                this.setMaxTreeDepth(best.maxTreeDepth);
                this.setNumIterations(best.numIterations);
                trainResults=best.results;
            }else//Not all present, just ditch
                System.out.println(resultsPath+" error: missing  ="+missing+" parameter values");
        }
        else{
            printlnDebug("\nTies Handling: ");
            for (XGBoostParamResultsHolder tie : ties) {
                printlnDebug(tie.toString());
            }
            printlnDebug("\n");
            
//            XGBoostParamResultsHolder best=ties.get(rng.nextInt(ties.size()));
            XGBoostParamResultsHolder best=Collections.max(ties); //get the most conservative (see XGBoostParamResultsHolder.computeconservedness())
            printlnDebug("Best learning rate ="+best.learningRate+" best max depth = "+best.maxTreeDepth+" best num iterations ="+best.numIterations+" acc = " + trainResults.acc + " (num ties = " + ties.size() + ")");
            
            this.setLearningRate(best.learningRate);
            this.setMaxTreeDepth(best.maxTreeDepth);
            this.setNumIterations(best.numIterations);
            trainResults=best.results;
         }     
    }
    
    /**
     * Does the 'actual' initialising and building of the model, as opposed to experimental code
     * setup etc
     * @throws Exception 
     */    
    public void buildActualClassifer() throws Exception {
        if(tuneParameters)
            tuneHyperparameters();
        
        String objective = "multi:softprob"; 
//        String objective = numClasses == 2 ? "binary:logistic" : "multi:softprob";
        
        trainDMat = wekaInstancesToDMatrix(trainInsts);
        params = new HashMap<String, Object>();
        //todo: this is a mega hack to enforce 1 thread only on cluster (else bad juju).
        //fix some how at some point. 
        if (runSingleThreaded || System.getProperty("os.name").toLowerCase().contains("linux"))
            params.put("nthread", 1);
        // else == num processors by default
        
        //fixed params
        params.put("silent", 1);
        params.put("objective", objective);
        if(objective.contains("multi"))
            params.put("num_class", numClasses); //required with multiclass problems
        params.put("seed", seed);
        params.put("subsample", rowSubsampling);
        params.put("colsample_bytree", colSubsampling);
        
        //tunable params (numiterations passed directly to XGBoost.train(...)
        params.put("learning_rate", learningRate);
        params.put("max_depth", maxTreeDepth);
        params.put("min_child_weight", minChildWeight);
        
        watches = new HashMap<String, DMatrix>();
//        if (getDebugPrinting() || getDebug())
//        watches.put("train", trainDMat);
        
//        int earlyStopping = (int) Math.ceil(numIterations / 10.0); 
        //e.g numIts == 25    =>   stop after 3 increases in err 
        //    numIts == 250   =>   stop after 25 increases in err
        
//        booster = XGBoost.train(trainDMat, params, numIterations, watches, null, null, null, earlyStopping);
        booster = XGBoost.train(trainDMat, params, numIterations, watches, null, null);
       
    }
    
    public ClassifierResults estimateTrainAcc(Instances insts) throws Exception {
        printlnDebug("estimateTrainAcc()");
        
        TunedXGBoost xg = new TunedXGBoost();
        xg.setLearningRate(learningRate);
        xg.setMaxTreeDepth(maxTreeDepth);
        xg.setMinChildWeight(minChildWeight);
        xg.setNumIterations(numIterations);
        xg.tuneParameters=false;
        xg.estimateAcc=false;
        xg.setSeed(seed);
        
        CrossValidator cv = new CrossValidator();
        cv.setSeed(seed); 
        cv.setNumFolds(cvFolds);
        cv.buildFolds(insts);
        
        return cv.crossValidateWithStats(xg, insts);
    }
    
    @Override
    public void buildClassifier(Instances insts) throws Exception {
//        long startTime=System.nanoTime(); 
        long startTime=System.currentTimeMillis(); 
        
        booster = null;
        trainResults =new ClassifierResults();
        
        trainInsts = new Instances(insts);
        numTrainInsts = insts.numInstances();
        numAtts = insts.numAttributes();
        numClasses = insts.numClasses();
        
        if(cvFolds>numTrainInsts)
            cvFolds=numTrainInsts;
//        rng = new Random(seed); //for tie resolution etc if needed
        
        buildActualClassifer();
        
        if(estimateAcc && !tuneParameters) //if tuneparas, will take the cv results of the best para set
            trainResults = estimateTrainAcc(trainInsts);
        
        if(saveEachParaAcc)
            trainResults.buildTime=combinedBuildTime;
        else
            trainResults.buildTime=System.currentTimeMillis()-startTime;
//            trainResults.buildTime=System.nanoTime()-startTime;
        if(trainPath!=""){  //Save basic train results
            OutFile f= new OutFile(trainPath);
            f.writeLine(trainInsts.relationName()+"," + (tuneParameters ? "TunedXGBoost" : "XGBoost") + ",Train");
            f.writeLine(getParameters());
            f.writeLine(trainResults.acc+"");
            f.writeLine(trainResults.writeInstancePredictions());
            f.closeFile();
        } 
    }

    @Override
    public double[] distributionForInstance(Instance inst) {
        double[] dist = new double[numClasses];
        
        //converting inst to dmat form
        Instances instHolder = new Instances(trainInsts, 0);
        instHolder.add(inst);
        DMatrix testInstMat = null;
        
        try {
             testInstMat = wekaInstancesToDMatrix(instHolder);
        } catch (XGBoostError ex) {
            System.err.println("Error converting test inst to DMatrix form: \n" + ex);
            System.exit(0);
        }
                
        //predicting, converting back to double[]
        try {
            float[][] predicts = booster.predict(testInstMat);
            for (int c = 0; c < numClasses; c++) 
                dist[c] = predicts[0][c];
        } catch (XGBoostError ex) {
            System.err.println("Error predicting test inst: \n" + ex);
            System.exit(0);
        }
        
        return dist;
    }
    
    public static DMatrix wekaInstancesToDMatrix(Instances insts) throws XGBoostError {
        int numRows = insts.numInstances();
        int numCols = insts.numAttributes()-1;
        
        float[] data = new float[numRows*numCols];
        float[] labels = new float[numRows];
        
        int ind = 0;
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++)
                data[ind++] = (float) insts.instance(i).value(j);
            labels[i] = (float) insts.instance(i).classValue();
        }
        
        DMatrix dmat = new DMatrix(data, numRows, numCols);
        dmat.setLabel(labels);
        return dmat;
    }
    
    
    /**
     * TrainAccuracyEstimate interface
     * @param string 
     */
    public void writeCVTrainToFile(String train) {
        trainPath=train;
        findTrainAcc=true;
    } 
 @Override
    public void setFindTrainAccuracyEstimate(boolean setCV){
        findTrainAcc=setCV;
    }
    
    /**
     * TrainAccuracyEstimate interface
     * @return 
     */
    @Override
     public ClassifierResults getTrainResults(){
        return trainResults;
    }  

    @Override
    public void setPathToSaveParameters(String r){
        resultsPath=r;
        setSaveEachParaAcc(true);
    }

    @Override
    public void setSaveEachParaAcc(boolean bln) {
        saveEachParaAcc=bln;
    }

    @Override
    public void setParamSearch(boolean bln) {
        tuneParameters=bln;
    }

    @Override
    public void setParametersFromIndex(int x) {
        tuneParameters=false;
        
        if(x<1 || x>numIterationsParaRange.length*learningRateParaRange.length*maxTreeDepthParaRange.length)//Error, invalid range
            throw new UnsupportedOperationException("ERROR parameter index "+x+" out of range for TunedXGBoost"); //To change body of generated methods, choose Tools | Templates.
        
        //x starts counting from 1 in parameter splittable for some reason, get it back to 0 in here
        x -= 1;
        
        int numIterationsIndex  = x % numIterationsParaRange.length;
        setNumIterations(numIterationsParaRange[numIterationsIndex]);
        x /= numIterationsParaRange.length;
        
        int maxTreeDepthIndex = x % maxTreeDepthParaRange.length;
        setMaxTreeDepth(maxTreeDepthParaRange[maxTreeDepthIndex]);
        x /= maxTreeDepthParaRange.length;
        
        int learningRateIndex = x;
        setLearningRate(learningRateParaRange[learningRateIndex]);
        
        
        printlnDebug("Index ="+x+" LearningRate="+learningRate+" MaxTreeDepth="+maxTreeDepth+" NumIterations ="+numIterations);
    }

    /**
     * SaveParameterInfo interface
     */
    @Override
    public String getParameters() {
        String result="BuildTime,"+trainResults.buildTime+",CVAcc,"+trainResults.acc;
        result+=",learningRate,"+learningRate;
        result+=",maxTreeDepth,"+maxTreeDepth;
        result+=",numIterations,"+numIterations;
        if (tuneParameters) {
            result+=",learningRateSpace,"+Arrays.toString(learningRateParaRange).replace(",", "/").replace(" ", "");
            result+=",maxTreeDepthSpace,"+Arrays.toString(maxTreeDepthParaRange).replace(",", "/").replace(" ", "");
            result+=",numIterationsSpace,"+Arrays.toString(numIterationsParaRange).replace(",", "/").replace(" ", "");
            for(double d:paramAccuracies)
                result+=","+d;
        }
        
        return result;
    }
    
    /**
     * ParameterSplittable interface
     */
    @Override
    public String getParas() {
        return getParameters();
    }

    @Override
    public double getAcc() {
        return trainResults.acc;
    }

    /**
     * Provides a smallish speedup when crossvalidating to tune hyperparameters. 
     * At current, will just speed up the search for the num iterations for a given set
     * of the other 3 params, storing the models built on each of the cv folds for a 
     * number of iterations, and continuing to build from those when evaluating higher number of iterations.
     * 
     * It's definitely imaginable in concept that this same process could be applied to the other params,
     * but would require going into the xgboost library code. nah. 
     * 
     * The spaghetti code is real.
     */
    private static class TuningXGBoostCrossValidationWrapper extends AbstractClassifier {

        final int numModels = 10;
        int modelIndex;
        TunedXGBoost[] models;
        
        float learningRate;
        int maxTreeDepth;
        int newNumIterations;
        int numIterations;
        
        public TuningXGBoostCrossValidationWrapper(float learningRate, int maxTreeDepth) {
            this.learningRate = learningRate;
            this.maxTreeDepth = maxTreeDepth;
            this.newNumIterations = 0;
            this.numIterations = 0;
            
            int modelIndex = 0;
            models = new TunedXGBoost[numModels];
            for (int i = 0; i < numModels; i++) {
                models[i] = new TunedXGBoost();
                models[i].setTuneParameters(false);
                models[i].estimateAcc = false;
                models[i].setLearningRate(learningRate);
                models[i].setMaxTreeDepth(maxTreeDepth);
                models[i].setNumIterations(newNumIterations);
            }
            
        }
        
        public void setSeed(int seed) {
            for (int i = 0; i < numModels; i++)
                models[i].setSeed(seed);
        }
        
        public void setNextNumIterations(int newNumIts) {
            numIterations = newNumIterations;
            newNumIterations = newNumIts;
            modelIndex = -1;
        }
        
        @Override
        public void buildClassifier(Instances data) throws Exception {
            //instead of (on a high level) calling build classifier on the same thing 10 times, 
            //with each subsequent call overwriting the training done in the last, 
            //we'll instead build each classifier in the models[] once, storing the traind model for each cv fold
            //when we move to the next num iterations, instead of building from scratch
            //we'll continue iterating from the stored models, which we can do since the 
            //cv folds will be identical.
            // so for a given para set, this build classifier will essentially be called 10 times,
            //once for each cv fold 
                
            modelIndex++; //going to use this model for this fold
            TunedXGBoost model = models[modelIndex];
            
            if (numIterations == 0) {
                //first of the 'numiterations' paras, i.e first build of each model. just build normally
                // - including the initialisation of all the meta info
                model.buildClassifier(data);
            } else {
                //continuing on from an already build model with less iterations
                //dont call normal build classifier, since that'll reinitialise 
                //a bunch of stuff, including the booster itself. instead just 
                //continue with a modified call to the trainer function
                model.booster = XGBoost.train(model.trainDMat, model.params, newNumIterations - numIterations, model.watches, null, null, null, 0, model.booster);
            }
        }
        
        @Override
        public double[] distributionForInstance(Instance inst) {
            return models[modelIndex].distributionForInstance(inst);
        }
    }
    

    
    public static void main(String[] args) throws Exception {
        
//        for (int fold = 0; fold < 15; fold++) { 
//            for (String dataset : DataSets.UCIContinuousFileNames) {
//                Experiments.main(new String[] { "Z:/Data/UCIContinuous/", "Z:/CawpeResubmissionDump/XGBoostTimingsForHESCA/", "true", "XGBoostSingleThread", dataset, ""+(fold+1) });
//            }
//        }
//        for (int fold = 15; fold < 30; fold++) { 
//            for (String dataset : DataSets.UCIContinuousFileNames) {
//                Experiments.main(new String[] { "Z:/Data/UCIContinuous/", "Z:/CawpeResubmissionDump/XGBoostTimingsForHESCA/", "true", "XGBoostSingleThread", dataset, ""+(fold+1) });
//            }
//        }
            
        //para split 
//        for (int para = 1; para <= 27; para++)
//            Experiments.main(new String[] { "Z:/Data/UCIDelgado/", "C:/Temp/XGBoostParaSplitTest/", "true", "TunedXGBoost", "hayes-roth", "1", "false", ""+para});
//        Experiments.main(new String[] { "Z:/Data/UCIDelgado/", "C:/Temp/XGBoostParaSplitTest/", "true", "TunedXGBoost", "hayes-roth", "1", "true"});
        //end para split 
        
        //checkpoint
//        Experiments.main(new String[] { "Z:/Data/UCIDelgado/", "C:/Temp/XGBoostCheckpointTest/", "true", "TunedXGBoost", "hayes-roth", "1", "true"});
        
        //standard
        Experiments.main(new String[] { "Z:/Data/UCIDelgado/", "C:/Temp/XGBoostStraightUpTest/", "true", "TunedXGBoost", "hayes-roth", "1", });

    }
    
    public static void listInvalidFiles(String base, StringBuilder sb){     
        File[] files = (new File(base)).listFiles();
        if (files.length == 0)
            return;
        
        for (File file : files) {
            if (file.isDirectory())
                listInvalidFiles(base + file.getName(), sb);
            else {
                try {
                    new ClassifierResults(file.getAbsolutePath());
                }catch (Exception e) {
                    System.out.println(file.getAbsolutePath());
//                    sb.append(file.getAbsolutePath()).append("\n");
                }
            }
        } 
    }
    
    public static void editTestFilesWithoutCorrespondingTrain() throws Exception {
        String path = "Z:\\Results\\TunedXGBoost\\Predictions\\";
        String safetyWritePath = "C:/Temp/XGBoostTestBackups/";
        
        
        for (String dataset : DataSets.UCIContinuousWithoutBigFour) {
            for (int fold = 0; fold < 30; fold++) {
                File trainFile = new File(path + dataset + "/trainFold" + fold + ".csv");
                try {
                    new ClassifierResults(trainFile.getAbsolutePath());
                } catch (Exception e) {
                    //couldnt load the train file, so edit the test file as missing so that experiments 
                    //can redo it
                    File testFile = new File(path + dataset + "/testFold" + fold + ".csv");
                    if (testFile.exists()) {
                        ClassifierResults tempRes = new ClassifierResults(testFile.getAbsolutePath());
                        (new File(safetyWritePath + dataset)).mkdirs();
                        OutFile out = new OutFile(safetyWritePath + dataset + "/testFold" + fold + ".csv");
                        out.writeString(tempRes.writeResultsFileToString());
                        out.closeFile();
                        
                        if (!testFile.renameTo(new File(testFile.getAbsolutePath().replace(".csv", "EDITNOTRAIN.csv"))))
                            throw new Exception("couldn't rename: " + testFile.getAbsolutePath());
                        else {
                            System.out.println("Edited: " + testFile.getAbsolutePath());
                        }
                    }
                }
            }
        }
    }
}
