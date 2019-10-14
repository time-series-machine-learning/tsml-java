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
package timeseriesweka.classifiers.interval_based;


import java.io.*;
import java.security.InvalidParameterException;
import java.util.ArrayList;

import timeseriesweka.classifiers.*;
import utilities.ClassifierTools;
import evaluation.evaluators.CrossValidationEvaluator;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.RandomTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;

import java.util.concurrent.TimeUnit;
import java.util.function.Function;

import weka.classifiers.Classifier;
import weka.core.Randomizable;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;

/**
 <!-- globalinfo-start -->
 * Implementation of Time Series Forest
 Time Series Forest (TimeSeriesForest) Deng 2013:
 * buildClassifier
 * Overview: Input n series length m
 * for each tree
 *      sample sqrt(m) intervals
 *      find three features on each interval: mean, standard deviation and slope
 *      concatenate to new feature set
 *      build tree on new feature set
 * classifyInstance
 *   ensemble the trees with majority vote

 * This implementation may deviate from the original, as it is using the same
 * structure as the weka random forest. In the paper the splitting criteria has a
 * tiny refinement. Ties in entropy gain are split with a further stat called margin
 * that measures the distance of the split point to the closest data.
 * So if the split value for feature
 * f=f_1,...f_n is v the margin is defined as
 *   margin= min{ |f_i-v| }
 * for simplicity of implementation, and for the fact when we did try it and it made
 * no difference, we have not used this. Note also, the original R implementation
 * may do some sampling of cases

 * Update 1:
 * * A few changes made to enable testing refinements.
 *1. general baseClassifier rather than a hard coded RandomTree. We tested a few
 *  alternatives, the summary results are available at
 *  http://www.timeseriesclassification.com/Experiments/TSF.xlsx
 *  Summary:
 *  Base Classifier:
 *       a) C.45 (J48) significantly worse than random tree
 *       b) CAWPE tbc
 *       c) CART tbc
 * 2. Added setOptions to allow parameter tuning. Tuning on parameters
 *       #trees, #features
 <!-- globalinfo-end -->
 <!-- technical-bibtex-start -->
 * Bibtex
 * <pre>
 * @article{deng13forest,
 * author = {H. Deng and G. Runger and E. Tuv and M. Vladimir},
 * title = {A time series forest for classification and feature extraction},
 * journal = {Information Sciences},
 * volume = {239},
 * year = {2013}
 *}
</pre>
<!-- technical-bibtex-end -->
<!-- options-start -->
 * Valid options are: <p/>
 *
 * <pre> -T
 *  set number of trees in the ensemble.</pre>
 *
 * <pre> -I
 *  set number of intervals to calculate.</pre>
<!-- options-end -->

 * @author ajb
 * @date 7/10/15
 * @update1 14/2/19

 **/

public class cTSF extends EnhancedAbstractClassifier
        implements TechnicalInformationHandler,
        TrainTimeContractable, Checkpointable {
//Static defaults

    private final static int DEFAULT_NUM_CLASSIFIERS=500;

    /** Primary parameters potentially tunable*/
    private int numClassifiers=DEFAULT_NUM_CLASSIFIERS;

    private int maxClassifiers = 1000;

    /** numIntervalsFinder sets numIntervals in buildClassifier. */
    private int numIntervals=0;
    transient Function<Integer,Integer> numIntervalsFinder = (numAtts) -> (int)(Math.sqrt(numAtts));
    /** Secondary parameter, mainly there to avoid single item intervals,
     which have no slope or std dev*/
    private int minIntervalLength=3;

    /** Ensemble members of base classifier, default to random forest RandomTree */
    private ArrayList<Classifier> trees;
    private Classifier base= new RandomTree();

    /** for each classifier [i]  interval j  starts at intervals[i][j][0] and
     ends  at  intervals[i][j][1] */
    private ArrayList<int[][]> intervals;

    /**Holding variable for test classification in order to retain the header info*/
    private Instances testHolder;


    /** If trainAccuracy is required, a cross validation is done in buildClassifier
     * or a OOB estimate is formed
     If set, train results are overwritten with each call to buildClassifier
     File opened on this path.*/
    boolean trainAccuracyEst=false;
    private String trainCVPath="";

    /** voteEnsemble determines whether to aggregate classifications or
     * probabilities when predicting */
    private boolean voteEnsemble=true;

    /** Flags and data required if Bagging **/
    private boolean bagging=false; //Use if we want an OOB estimate
    private ArrayList<boolean[]> inBag;
    private int[] oobCounts;
    private double[][] trainDistributions;

    private boolean checkpoint = false;
    private String checkpointPath;
    private long checkpointTime = 0;
    private long checkpointTimeElapsed= 0;

    private boolean trainTimeContract = false;
    private long contractTime = 0;
    
    protected static final long serialVersionUID = 32554L;

    public cTSF(){
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
    }
    public cTSF(int s){
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
        setSeed(seed);
    }
    /**
     *
     * @param c a base classifier constructed elsewhere and cloned into ensemble
     */
    public void setBaseClassifier(Classifier c){
        base=c;
    }
    public void setBagging(boolean b){
        bagging=b;
    }

    /**
     * ok,  two methods are a bit pointless, experimenting with ensemble method
     * @param b
     */
    public void setVoteEnsemble(boolean b){
        voteEnsemble=b;
    }
    public void setProbabilityEnsemble(boolean b){
        voteEnsemble=!b;
    }
    
    /**
     * Perhaps make this coherent with setOptions(String[] ar)?
     * @return String written to results files
     */
    @Override
    public String getParameters() {
        String temp=super.getParameters()+",numTrees,"+numClassifiers+",numIntervals,"+numIntervals+",voting,"+voteEnsemble+",BaseClassifier,"+base.getClass().getSimpleName()+",Bagging,"+bagging;
        if(base instanceof RandomTree)
            temp+=",AttsConsideredPerNode,"+((RandomTree)base).getKValue();
        return temp;

    }
    public void setNumTrees(int t){
        numClassifiers=t;
    }


    //<editor-fold defaultstate="collapsed" desc="results reported in Info Sciences paper">
    static double[] reportedResults={
            0.2659,
            0.2302,
            0.2333,
            0.0256,
            0.2537,
            0.0391,
            0.0357,
            0.2897,
            0.2,
            0.2436,
            0.049,
            0.08,
            0.0557,
            0.2325,
            0.0227,
            0.101,
            0.1543,
            0.0467,
            0.552,
            0.6818,
            0.0301,
            0.1803,
            0.2603,
            0.0448,
            0.2237,
            0.119,
            0.0987,
            0.0865,
            0.0667,
            0.4339,
            0.233,
            0.1868,
            0.0357,
            0.1056,
            0.1116,
            0.0267,
            0.02,
            0.1177,
            0.0543,
            0.2102,
            0.2876,
            0.2624,
            0.0054,
            0.3793,
            0.1513
    };
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="problems used in Info Sciences paper">
    static String[] problems={
            "FiftyWords",
            "Adiac",
            "Beef",
            "CBF",
            "ChlorineConcentration",
            "CinCECGtorso",
            "Coffee",
            "CricketX",
            "CricketY",
            "CricketZ",
            "DiatomSizeReduction",
            "ECG",
            "ECGFiveDays",
            "FaceAll",
            "FaceFour",
            "FacesUCR",
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
            "NonInvasiveFetalECGThorax1",
            "NonInvasiveFetalECGThorax2",
            "OliveOil",
            "OSULeaf",
            "SonyAIBORobotSurface1",
            "SonyAIBORobot Surface2",
            "StarLightCurves",
            "SwedishLeaf",
            "Symbols",
            "Synthetic Control",
            "Trace",
            "TwoLeadECG",
            "TwoPatterns",
            "UWaveGestureLibraryX",
            "UWaveGestureLibraryY",
            "UWaveGestureLibraryZ",
            "Wafer",
            "WordsSynonyms",
            "Yoga"
    };
    //</editor-fold>

    /**
     * paper defining TSF
     * @return TechnicalInformation
     */
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "H. Deng, G. Runger, E. Tuv and M. Vladimir");
        result.setValue(TechnicalInformation.Field.YEAR, "2013");
        result.setValue(TechnicalInformation.Field.TITLE, "A time series forest for classification and feature extraction");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Information Sciences");
        result.setValue(TechnicalInformation.Field.VOLUME, "239");
        result.setValue(TechnicalInformation.Field.PAGES, "142-153");

        return result;
    }


    /**
     * main buildClassifier
     * @param data
     * @throws Exception
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        // can classifier handle the data?
        getCapabilities().testWithFail(data);
        long t1=System.nanoTime();
        File file = new File(checkpointPath + "TSF" + seed + ".ser");
        //if checkpointing and serialised files exist load said files
        if (checkpoint && file.exists()){
        //path checkpoint files will be saved to
            if(debug)
               System.out.println("Loading from checkpoint file");
            loadFromFile(checkpointPath + "TSF" + seed + ".ser");
 //               checkpointTimeElapsed -= System.nanoTime()-t1;
        }
        //initialise variables
        else {
            numIntervals=numIntervalsFinder.apply(data.numAttributes()-1);
            //Estimate train accuracy here if required and not using bagging
//Set up instances size and format.
            trees=new ArrayList(numClassifiers);

            /** Set up for train rstimate **/
            if(trainAccuracyEst) {
                trainDistributions= new double[data.numInstances()][data.numClasses()];
            }

            /** Set up for Bagging **/
            if(bagging){
                inBag=new ArrayList();
                oobCounts=new int[data.numInstances()];
            }

            //cancel loop using time instead of number built.
            if (trainTimeContract){
                numClassifiers = 0;
            }

            intervals = new ArrayList();
        }

        ArrayList<Attribute> atts=new ArrayList<>();
        String name;
        for(int j=0;j<numIntervals*3;j++){
            name = "F"+j;
            atts.add(new Attribute(name));
        }
        //Get the class values as an array list
        Attribute target =data.attribute(data.classIndex());
        ArrayList<String> vals=new ArrayList<>(target.numValues());
        for(int j=0;j<target.numValues();j++)
            vals.add(target.value(j));
        atts.add(new Attribute(data.attribute(data.classIndex()).name(),vals));
        //create blank instances with the correct class value
        Instances result = new Instances("Tree",atts,data.numInstances());
        result.setClassIndex(result.numAttributes()-1);
        for(int i=0;i<data.numInstances();i++){
            DenseInstance in=new DenseInstance(result.numAttributes());
            in.setValue(result.numAttributes()-1,data.instance(i).classValue());
            result.add(in);
        }

        testHolder =new Instances(result,0);
        DenseInstance in=new DenseInstance(result.numAttributes());
        testHolder.add(in);
//Need to hard code this because log(m)+1 is sig worse than sqrt(m) is worse than using all!
        if(base instanceof RandomTree){
            ((RandomTree) base).setKValue(result.numAttributes()-1);
//            ((RandomTree) base).setKValue((int)Math.sqrt(result.numAttributes()-1));
        }

        int classifiersBuilt = trees.size();

        /** For each base classifier
         *      generate random intervals
         *      do the transfrorms
         *      build the classifier
         * */
        while(((System.nanoTime()-t1)+checkpointTimeElapsed < contractTime || classifiersBuilt < numClassifiers)
            && classifiersBuilt < maxClassifiers){
            //1. Select random intervals for tree i
            int[][] interval =new int[numIntervals][2];  //Start and end
            if(data.numAttributes()-1<minIntervalLength)
                minIntervalLength=data.numAttributes()-1;
            for(int j=0;j<numIntervals;j++){
                interval[j][0]=rand.nextInt(data.numAttributes()-1-minIntervalLength);       //Start point
                int length=rand.nextInt(data.numAttributes()-1-interval[j][0]);//Min length 3
                if(length<minIntervalLength)
                    length=minIntervalLength;
                interval[j][1]=interval[j][0]+length;
            }
            //2. Generate and store attributes
            for(int j=0;j<numIntervals;j++){
                //For each instance
                for(int k=0;k<data.numInstances();k++){
                    //extract the interval
                    double[] series=data.instance(k).toDoubleArray();
                    FeatureSet f= new FeatureSet();
                    f.setFeatures(series, interval[j][0], interval[j][1]);
                    result.instance(k).setValue(j*3, f.mean);
                    result.instance(k).setValue(j*3+1, f.stDev);
                    result.instance(k).setValue(j*3+2, f.slope);
                }
            }
            //3. Create and build tree using all the features. Feature selection
            Classifier tree = AbstractClassifier.makeCopy(base);
            if(seedClassifier && tree instanceof Randomizable)
                ((Randomizable)tree).setSeed(seed*(classifiersBuilt+1));
            if(bagging){
                boolean[] bag = new boolean[result.numInstances()];
                Instances bagData = result.resampleWithWeights(rand, bag);
                tree.buildClassifier(bagData);
                if(trainAccuracyEst){
                    for(int j=0;j<result.numInstances();j++){
                        if(bag[j])
                            continue;
                        double[] newProbs = tree.distributionForInstance(result.instance(j));
                        oobCounts[j]++;
                        for(int k=0;k<newProbs.length;k++)
                            trainDistributions[j][k]+=newProbs[k];

                    }
                }
                inBag.add(bag);
            }
            else{
                tree.buildClassifier(result);

                if(trainAccuracyEst) {
                    /** Defaults to 10 or numInstances, whichever is smaller.
                     * Interface TrainAccuracyEstimate
                     * Could this be handled better? */
                    int numFolds = setNumberOfFolds(data);
                    CrossValidationEvaluator cv = new CrossValidationEvaluator();
                    if (seedClassifier)
                        cv.setSeed(seed);
                    cv.setNumFolds(numFolds);
                    ClassifierResults results = cv.crossValidateWithStats(AbstractClassifier.makeCopy(base), result);

                    for (int j = 0; j < result.numInstances(); j++) {
                        double[] newProbs = results.getProbabilityDistribution(j);
                        for (int k = 0; k < newProbs.length; k++)
                            trainDistributions[j][k] += newProbs[k];
                    }
                }
            }

            intervals.add(interval);
            trees.add(tree);
            classifiersBuilt++;

            if (checkpoint){
                checkpoint(t1);
            }
        }

        long t2=System.nanoTime();
        //Store build time, this is always recorded
        trainResults.setBuildTime((t2-t1)+checkpointTimeElapsed);
        //If trainAccuracyEst ==true and we want to save results, write out object
        if(trainAccuracyEst){
            if(!bagging){//Do a CV
                double[] preds=new double[data.numInstances()];
                double[] actuals=new double[data.numInstances()];
                for(int j=0;j<data.numInstances();j++){
                    for(int k=0;k<trainDistributions[j].length;k++)
                        trainDistributions[j][k]/=trees.size();
                    preds[j]=utilities.GenericTools.indexOfMax(trainDistributions[j]);
                    actuals[j]=data.instance(j).classValue();
                }
                long[] predTimes=new long[data.numInstances()];//Dummy variable, need something
                trainResults.addAllPredictions(preds, trainDistributions, predTimes, null);
                trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
                trainResults.setClassifierName("TSF");
                trainResults.setDatasetName(data.relationName());
                trainResults.setSplit("train");
                trainResults.setFoldID(seed);
                trainResults.setParas(getParameters());
                trainResults.finaliseResults(actuals);

//                /** Defaults to 10 or numInstances, whichever is smaller.
//                 * Interface TrainAccuracyEstimate
//                 * Could this be handled better? */
//                int numFolds=setNumberOfFolds(data);
//                CrossValidationEvaluator cv = new CrossValidationEvaluator();
//                if (setSeed)
//                    cv.setSeed(seed);
//                cv.setNumFolds(numFolds);
//                TSF tsf=new TSF();
//                tsf.setFindTrainAccuracyEstimate(false);
//                trainResults=cv.crossValidateWithStats(tsf,data);
            }else{
                // Use bag data. Normalise probs
                double[] preds=new double[data.numInstances()];
                double[] actuals=new double[data.numInstances()];
                for(int j=0;j<data.numInstances();j++){
                    for(int k=0;k<trainDistributions[j].length;k++)
                        trainDistributions[j][k]/=oobCounts[j];
                    preds[j]=utilities.GenericTools.indexOfMax(trainDistributions[j]);
                    actuals[j]=data.instance(j).classValue();
                }
                long[] predTimes=new long[data.numInstances()];//Dummy variable, need something
                trainResults.addAllPredictions(preds, trainDistributions, predTimes, null);
                trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
                trainResults.setClassifierName("TSFBagging");
                trainResults.setDatasetName(data.relationName());
                trainResults.setSplit("train");
                trainResults.setFoldID(seed);
                trainResults.setParas(getParameters());
                trainResults.finaliseResults(actuals);
            }
            if(trainCVPath!=""){
                trainResults.writeFullResultsToFile(trainCVPath);
/*                OutFile of=new OutFile(trainCVPath);
                of.writeLine(data.relationName()+",TSF,train");
                of.writeLine(getParameters());
               of.writeLine(trainResults.getAcc()+"");
               double[] trueClassVals,predClassVals;
               trueClassVals=trainResults.getTrueClassValsAsArray();
               predClassVals=trainResults.getPredClassValsAsArray();
               for(int i=0;i<data.numInstances();i++){
                   //Basic sanity check
                   if(data.instance(i).classValue()!=trueClassVals[i]){
                       throw new Exception("ERROR in TSF cross validation, class mismatch!");
                   }
                   of.writeString((int)trueClassVals[i]+","+(int)predClassVals[i]+",");
                   for(double d:trainResults.getProbabilityDistribution(i))
                       of.writeString(","+d);
                   of.writeString("\n");
               }
*/
            }
        }

    }
    /**
     * Sums either the
     * @param ins to classifier
     * @return array of doubles: probability of each class
     * @throws Exception
     */
    @Override
    public double[] distributionForInstance(Instance ins) throws Exception {
        double[] d=new double[ins.numClasses()];
        //Build transformed instance
        double[] series=ins.toDoubleArray();
        for(int i=0;i<trees.size();i++){
            for(int j=0;j<numIntervals;j++){
                //extract all intervals
                FeatureSet f= new FeatureSet();
                f.setFeatures(series, intervals.get(i)[j][0], intervals.get(i)[j][1]);
                testHolder.instance(0).setValue(j*3, f.mean);
                testHolder.instance(0).setValue(j*3+1, f.stDev);
                testHolder.instance(0).setValue(j*3+2, f.slope);
            }
            if(voteEnsemble){
                int c=(int)trees.get(i).classifyInstance(testHolder.instance(0));
                d[c]++;
            }else{
                double[] temp=trees.get(i).distributionForInstance(testHolder.instance(0));
                for(int j=0;j<temp.length;j++)
                    d[j]+=temp[j];
            }
        }
        double sum=0;
        for(double x:d)
            sum+=x;
        for(int i=0;i<d.length;i++)
            d[i]=d[i]/sum;
        return d;
    }
    /**
     * What about
     * @param ins
     * @return
     * @throws Exception
     */
    @Override
    public double classifyInstance(Instance ins) throws Exception {
        double[] d=distributionForInstance(ins);
        int max=0;
        for(int i=1;i<d.length;i++)
            if(d[i]>d[max])
                max=i;
        return (double)max;
    }
    /**
     * Parses a given list of options to set the parameters of the classifier.
     * We use this for the tuning mechanism, setting parameters through setOptions
     <!-- options-start -->
     * Valid options are: <p/>
     * <pre> -T
     * Number of trees.</pre>
     *
     * <pre> -I
     * Number of intervals to fit.</pre>
     *
     <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception{
        System.out.print("TSF para sets ");
        for (String str:options)
            System.out.print(","+str);
        System.out.print("\n");
        String numTreesString=Utils.getOption('T', options);
        if (numTreesString.length() != 0)
            numClassifiers = Integer.parseInt(numTreesString);
        else
            numClassifiers = DEFAULT_NUM_CLASSIFIERS;

        String numFeaturesString=Utils.getOption('I', options);
//Options here are a double between 0 and 1 (proportion of features), a text
//string sqrt or log, or an integer number
        if (numTreesString.length() != 0){
            try{
                if(numFeaturesString.equals("sqrt"))
                    numIntervalsFinder = (numAtts) -> (int)(Math.sqrt(numAtts));
                else if(numFeaturesString.equals("log"))
                    numIntervalsFinder = (numAtts) -> (int) Utils.log2(numAtts) + 1;
                else{
                    double d=Double.parseDouble(numFeaturesString);
                    if(d<=0)
                        throw new Exception("proportion of features of of range 0 to 1");
                    if(d<=1)
                        numIntervalsFinder = (numAtts) -> (int)(d*numAtts);
                    else
                        numIntervalsFinder = (numAtts) -> (int)(d);

//                        System.out.println("Proportion/number of intervals = "+d);
                }
            }catch(Exception e){
                System.err.print(" Error: invalid parameter passed to TSF setOptions for number of parameters. Setting to default");
                System.err.print("Value"+numIntervalsFinder+" Permissable values: sqrt, log, or a double range 0...1");
                numIntervalsFinder = (numAtts) -> (int)(Math.sqrt(numAtts));
            }
        }
        else
            System.out.println("Unable to read number of intervals, not set");
    }

    @Override //Checkpointable
    public boolean setSavePath(String path) {
        boolean validPath=Checkpointable.super.setSavePath(path);
        if(validPath){
            checkpointPath = path;
            checkpoint = true;
        }
        return validPath;
    }

    @Override
    public void copyFromSerObject(Object obj) throws Exception {
        if(!(obj instanceof cTSF))
            throw new Exception("The SER file is not an instance of TSF");
        cTSF saved = ((cTSF)obj);
        System.out.println("Loading TSF" + seed + ".ser");

        try{
            numClassifiers = saved.numClassifiers;
            maxClassifiers = saved.maxClassifiers;
            numIntervals = saved.numIntervals;
            //numIntervalsFinder = saved.numIntervalsFinder;
            minIntervalLength = saved.minIntervalLength;
            trees = saved.trees;
            base = saved.base;
            intervals = saved.intervals;
            //testHolder = saved.testHolder;
            rand = saved.rand;
            seedClassifier = saved.seedClassifier;
            seed = saved.seed;
            trainAccuracyEst = saved.trainAccuracyEst;
            trainCVPath = saved.trainCVPath;
            voteEnsemble = saved.voteEnsemble;
            bagging = saved.bagging;
            inBag = saved.inBag;
            oobCounts = saved.oobCounts;
            trainDistributions = saved.trainDistributions;
            //checkpoint = saved.checkpoint;
            //checkpointPath = saved.checkpointPath
            checkpointTime = saved.checkpointTime;
            checkpointTimeElapsed = saved.checkpointTime; //intentional, time spent building previously unchanged
            trainTimeContract = saved.trainTimeContract;
            contractTime = saved.contractTime;
        }catch(Exception ex){
            System.out.println("Unable to assign variables when loading serialised file");
        }
    }

    @Override
    public void setTrainTimeLimit(TimeUnit time, long amount) {
        switch (time){
            case DAYS:
                contractTime = (long)(8.64e+13)*amount;
                break;
            case HOURS:
                contractTime = (long)(3.6e+12)*amount;
                break;
            case MINUTES:
                contractTime = (long)(6e+10)*amount;
                break;
            case SECONDS:
                contractTime = (long)(1e+9)*amount;
                break;
            case NANOSECONDS:
                contractTime = amount;
                break;
            default:
                throw new InvalidParameterException("Invalid time unit");
        }
        trainTimeContract = true;
    }

    private void checkpoint(long startTime){
        if(checkpointPath!=null){
            try{
                long t1 = System.nanoTime();
                File f = new File(checkpointPath);
                if(!f.isDirectory())
                    f.mkdirs();

                //time spent building so far.
                checkpointTime = ((System.nanoTime() - startTime) + checkpointTimeElapsed);

                //save this, classifiers and train data not included
                saveToFile(checkpointPath + "TSF" + seed + "temp.ser");

                File file = new File(checkpointPath + "TSF" + seed + "temp.ser");
                File file2 = new File(checkpointPath + "TSF" + seed + ".ser");
                file2.delete();
                file.renameTo(file2);

                checkpointTimeElapsed -= System.nanoTime()-t1;
            }
            catch(Exception e){
                e.printStackTrace();
                System.out.println("Serialisation to "+checkpointPath+"TSF" + seed + ".ser FAILED");
            }
        }
    }

    //Nested class to store three simple summary features used to construct train data
    public static class FeatureSet{
        public static boolean findSkew=false;
        public static boolean findKurtosis=false;
        double mean;
        double stDev;
        double slope;
        double skew;
        double kurtosis;
        public void setFeatures(double[] data, int start, int end){
            double sumX=0,sumYY=0;
            double sumY3=0,sumY4=0;
            double sumY=0,sumXY=0,sumXX=0;
            int length=end-start+1;
            for(int i=start;i<=end;i++){
                sumY+=data[i];
                sumYY+=data[i]*data[i];
                sumX+=(i-start);
                sumXX+=(i-start)*(i-start);
                sumXY+=data[i]*(i-start);
            }
            mean=sumY/length;
            stDev=sumYY-(sumY*sumY)/length;
            slope=(sumXY-(sumX*sumY)/length);
            double denom=sumXX-(sumX*sumX)/length;
            if(denom!=0)
                slope/=denom;
            else
                slope=0;
            stDev/=length;
            if(stDev==0)    //Flat line
                slope=0;
//            else //Why not doing this? Because not needed? 
//                stDev=Math.sqrt(stDev);
            if(slope==0)
                stDev=0;
            if(findSkew){
                if(stDev==0)
                    skew=1;
                else{
                    for(int i=start;i<=end;i++)
                        sumY3+=data[i]*data[i]*data[i];
                    skew=sumY3-3*sumY*sumYY+2*sumY*sumY;
                    skew/=length*stDev*stDev*stDev;
                }
            }
            if(findKurtosis){
                if(stDev==0)
                    kurtosis=1;
                else{
                    for(int i=start;i<=end;i++)
                        sumY4+=data[i]*data[i]*data[i]*data[i];
                    kurtosis=sumY4-4*sumY*sumY3+6*sumY*sumY*sumYY-3*sumY*sumY*sumY*sumY;
                    skew/=length*stDev*stDev*stDev*stDev;
                }
            }

        }
        public void setFeatures(double[] data){
            setFeatures(data,0,data.length-1);
        }
        @Override
        public String toString(){
            return "mean="+mean+" stdev = "+stDev+" slope ="+slope;
        }
    }

    public static void main(String[] arg) throws Exception{
// Basic correctness tests, including setting paras through 
        String dataLocation="Z:\\Data\\TSCProblems2018\\";
        String resultsLocation="C:\\temp\\";
        String problem="ItalyPowerDemand";
        File f= new File(resultsLocation+problem);
        if(!f.isDirectory())
            f.mkdirs();
        Instances train=DatasetLoading.loadDataNullable(dataLocation+problem+"\\"+problem+"_TRAIN");
        Instances test=DatasetLoading.loadDataNullable(dataLocation+problem+"\\"+problem+"_TEST");
        cTSF tsf = new cTSF();
        tsf.setSeed(0);
        tsf.setTrainTimeLimit((long)1.5e+10);
        tsf.setSavePath("C:\\temp\\");
        tsf.setEstimateOwnPerformance(true);
        double a;
        tsf.buildClassifier(train);
        ClassifierResults trainres = tsf.getTrainResults();
        trainres.writeFullResultsToFile(resultsLocation+problem+"trainFold0.csv");
        System.out.println("build ok: original atts="+(train.numAttributes()-1)+" new atts ="+tsf.testHolder.numAttributes()+" num trees = "+tsf.numClassifiers+" num intervals = "+tsf.numIntervals);
        System.out.println(tsf.trainResults.getBuildTime());
        a=ClassifierTools.accuracy(test, tsf);
        System.out.println("Test Accuracy ="+a);
        System.out.println();

        tsf = new cTSF();
        tsf.setSeed(1);
        tsf.setTrainTimeLimit((long)1.5e+10);
        tsf.setSavePath("C:\\temp\\");
        String[] options=new String[4];
        options[0]="-T";
        options[1]="10";
        options[2]="-I";
        options[3]="1";
        tsf.setOptions(options);
        tsf.buildClassifier(train);
        System.out.println("build ok: original atts="+(train.numAttributes()-1)+" new atts ="+tsf.testHolder.numAttributes()+" num trees = "+tsf.numClassifiers+" num intervals = "+tsf.numIntervals);
        System.out.println(tsf.trainResults.getBuildTime());
        a=ClassifierTools.accuracy(test, tsf);
        System.out.println("Test Accuracy ="+a);


    }
}
