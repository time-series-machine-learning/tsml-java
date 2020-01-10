
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
package tsml.classifiers.interval_based;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.tuning.ParameterSpace;
import experiments.data.DatasetLoading;
import tsml.classifiers.Checkpointable;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.Tuneable;
import tsml.classifiers.hybrids.Catch22Classifier;
import tsml.transformers.Catch22;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomTree;
import weka.core.*;

import java.io.File;
import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

import static utilities.ClusteringUtilities.zNormalise;
import static utilities.ClusteringUtilities.zNormaliseWithClass;

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
 *  alternatives, the summary results NEED WRITING UP
 *  Summary:
 *  Base Classifier:
 *       a) C.45 (J48) significantly worse than random tree.
 *       b) CAWPE tbc
 *       c) CART tbc
 * 2. Added setOptions to allow parameter tuning. Tuning on parameters
 *       #trees, #features
 <!-- globalinfo-end -->
 <!-- technical-bibtex-start -->
 * Bibtex
 * <pre>
 * article{deng13forest,
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

 * author ajb
 * date 7/10/15
 * update1 14/2/19
 * update2 13/9/19: Adjust to allow three methods for estimating test accuracy

 **/

public class Catch22TSF extends EnhancedAbstractClassifier
        implements TechnicalInformationHandler, Tuneable, TrainTimeContractable, Checkpointable {
//Static defaults


    public int experimentalOptions = 0;
    public int intSelection = 0;
    public boolean norm = false;
    public boolean preNorm = false;
    public boolean outlierNorm = false;
    public boolean attSubsample = false;
    public boolean attSubsampleInterval = false;
    public int attSubsampleNum = 16;

    public int numAttributes = 22;
    public int ogNumAttributes;
    public ArrayList<ArrayList<Integer>> subsampleAtts;
    public ArrayList<ArrayList<Integer>>[] subsampleIntervalAtts;

    public int gsNumIntervals = 0;
    public int gsNumTrees = 0;
    public int gsRandomTreeK = 0;
    public int gsAttSubsamplePerTree = 0;
    public int gsBagging = 0;


    private final static int DEFAULT_NUM_CLASSIFIERS=500;

    /** Primary parameters potentially tunable*/
    private int numClassifiers=DEFAULT_NUM_CLASSIFIERS;

    /** numIntervalsFinder sets numIntervals in buildClassifier. */
    private int numIntervals=0;
    private Function<Integer,Integer> numIntervalsFinder = (numAtts) -> (int)(Math.sqrt(numAtts));
    /** Secondary parameter, mainly there to avoid single item intervals,
     which have no slope or std dev*/
    public int minIntervalLength=3;

    /** Ensemble members of base classifier, default to random forest RandomTree */
    private ArrayList<Classifier> trees;
    public Classifier base= new RandomTree();

    /** for each classifier [i]  interval j  starts at intervals[i][j][0] and
     ends  at  intervals[i][j][1] */
    private  ArrayList<int[][]> intervals;

    /**Holding variable for test classification in order to retain the header info*/
    private Instances testHolder;


    /** voteEnsemble determines whether to aggregate classifications or
     * probabilities when predicting */
    private boolean voteEnsemble=true;

    /** Flags and data required if Bagging **/
    private boolean bagging=false; //Use if we want an OOB estimate
    private boolean[][] inBag;
    private int[] oobCounts;
    private double[][] trainDistributions;

    /** If trainAccuracy is required, there are three mechanisms to obtain it:
     * 1. bagging == true: use the OOB accuracy from the final model
     * 2. bagging == false,estimator=CV: do a 10x CV on the train set with a clone
     * of this classifier
     * 3. bagging == false,estimator=OOB: build an OOB model just to get the OOB
     * accuracy estimate
     */
//    boolean findTrainPredictions=false;
    enum EstimatorMethod{CV,OOB}
    private EstimatorMethod estimator= EstimatorMethod.CV;
    private String trainFoldPath="";
/* If trainFoldPath is set, train results are overwritten with
 each call to buildClassifier.*/

    private boolean checkpoint = false;
    private String checkpointPath;
    private long checkpointTime = 0;
    private long checkpointTimeElapsed= 0;

    private boolean trainTimeContract = false;
    private long contractTime = 0;
    private int maxClassifiers = 500;

    protected static final long serialVersionUID = 222555L;


    public Catch22TSF(){
//TSF Has the capability to form train estimates
//In order to do this,
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
    }
    public Catch22TSF(int s){
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
        setSeed(s);
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
     * @param b boolean to set vote ensemble
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
        int nt = numClassifiers;
        if (trees != null) nt = trees.size();
        String temp=super.getParameters()+",numTrees,"+nt+",numIntervals,"+numIntervals+",voting,"+voteEnsemble+",BaseClassifier,"+base.getClass().getSimpleName()+",Bagging,"+bagging;
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
        TechnicalInformation    result;
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
    public void buildClassifier(Instances orgData) throws Exception {
/** Build Stage: 
 *  Builds the final classifier with or without bagging.  
 */
        // can classifier handle the data?
        getCapabilities().testWithFail(orgData);
        long t1=System.nanoTime();

        Instances data;
        if (preNorm){
            data = new Instances(orgData);
            zNormaliseWithClass(data);
        }
        else{
            data = orgData;
        }

        Catch22 c22 = new Catch22();

        File file = new File(checkpointPath + "TSF" + seed + ".ser");
        //if checkpointing and serialised files exist load said files
        if (checkpoint && file.exists()){
            //path checkpoint files will be saved to
            if(debug)
                System.out.println("Loading from checkpoint file");
            loadFromFile(checkpointPath + "TSF" + seed + ".ser");
        }
        //initialise variables
        else {

            //bagging
            //reduction of num intervals
            //random tree k change
            //random catch22 attribute subsample
            //attribute subsample norm+nonorm
            //normalisation of intervals
            //c4.5
            //windows per tree random length random overlap
            //oob cawpe weighting classic

//        minIntervalLength = 12;
//
//        if (minIntervalLength > (data.numAttributes()-1)/numIntervals){
//            minIntervalLength = (data.numAttributes()-1)/numIntervals;
//        }

            if (minIntervalLength < 1){
                minIntervalLength = (data.numAttributes()-1)/20;

                if (minIntervalLength < 3){
                    minIntervalLength = 3;
                }
            }

            if (outlierNorm) {
                c22.setOutlierNormalise(true);
            }

            switch (experimentalOptions) {
                case 1:
                    numIntervals = numIntervalsFinder.apply(data.numAttributes() - 1);
                    //numIntervals =  (int)(Math.sqrt(data.numAttributes()-1)/2);
                    //numIntervals =  (int)Math.round(Math.pow(Math.sqrt(data.numAttributes()-1), 0.85));
                    intSelection = 1;
                    break;
                case 2:
                    //+basic stats
                    numIntervals = numIntervalsFinder.apply(data.numAttributes() - 1);
                    numAttributes = 25;
                    break;
                case 3:
                    //+basic stats fast
                    numIntervals = (int) Math.round(Math.pow(Math.sqrt(data.numAttributes() - 1), 0.85));
                    numClassifiers = 250;
                    numAttributes = 25;
                    break;
                case 99:
                    if (gsNumIntervals == 0) {
                        numIntervals = numIntervalsFinder.apply(data.numAttributes() - 1);
                    } else if (gsNumIntervals == 1) {
                        numIntervals = (int) Math.round(Math.pow(Math.sqrt(data.numAttributes() - 1), 0.85));
                    } else if (gsNumIntervals == 2) {
                        numIntervals = (int) (Math.sqrt(data.numAttributes() - 1) / 2);
                    }

                    if (gsNumTrees == 0) {
                        numClassifiers = 500;
                    } else if (gsNumTrees == 1) {
                        numClassifiers = 250;
                    } else if (gsNumTrees == 2) {
                        numClassifiers = 100;
                    }

                    if (gsAttSubsamplePerTree == 0) {
                        attSubsample = false;
                    } else if (gsAttSubsamplePerTree == 1) {
                        attSubsample = true;
                        attSubsampleNum = 16;
                    } else if (gsAttSubsamplePerTree == 2) {
                        attSubsample = true;
                        attSubsampleNum = 10;
                    }

                    if (gsBagging == 0) {
                        bagging = false;
                    } else if (gsBagging == 1) {
                        bagging = true;
                    }
                    break;
                case 101:
                case 102:
                case 103:
                case 104:
                case 105:
                case 106:
                case 107:
                case 108:
                case 109:
                case 110:
                case 111:
                case 112:
                case 113:
                case 114:
                case 115:
                case 116:
                case 118:
                case 119:
                case 120:
                case 121:
                case 122:
                    //leave one feature out
                    numAttributes = 21;
                    numIntervals = numIntervalsFinder.apply(data.numAttributes() - 1);
                    break;
                default:
                    numIntervals = numIntervalsFinder.apply(data.numAttributes() - 1);
                    break;
            }

            if (attSubsample) {
                subsampleAtts = new ArrayList();

                if (attSubsampleNum < numAttributes) {
                    ogNumAttributes = numAttributes;
                    numAttributes = attSubsampleNum;
                }
            }
            else if (attSubsampleInterval){
                subsampleIntervalAtts = new ArrayList[numIntervals];

                for (int i = 0; i < numIntervals; i++){
                    subsampleIntervalAtts[i] = new ArrayList<>();
                }

                if (attSubsampleNum < numAttributes) {
                    ogNumAttributes = numAttributes;
                    numAttributes = attSubsampleNum;
                }
            }

//Set up instances size and format.

            /** Set up for Bagging if required **/
            if(bagging){
                inBag=new boolean[numClassifiers][data.numInstances()];
                oobCounts=new int[data.numInstances()];

                if (getEstimateOwnPerformance()){
                    trainDistributions = new double[data.numInstances()][data.numClasses()];
                }
            }

            //cancel loop using time instead of number built.
            if (trainTimeContract){
                numClassifiers = 0;
                trees = new ArrayList<>();
                intervals = new ArrayList<>();
            }
            else{
                trees = new ArrayList<>(numClassifiers);
                intervals = new ArrayList<>(numClassifiers);
            }

        }

        ArrayList<Attribute> atts=new ArrayList<>();
        String name;
        for(int j=0;j<numIntervals*numAttributes;j++){
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

            if (experimentalOptions == 99){
                if (gsRandomTreeK == 0){
                    ((RandomTree) base).setKValue(result.numAttributes() - 1);
                }
                else if (gsRandomTreeK == 1){
                    ((RandomTree) base).setKValue((int)Math.sqrt(result.numAttributes()-1));
                }
            }
            else {
                ((RandomTree) base).setKValue(result.numAttributes() - 1);
//            ((RandomTree) base).setKValue((int)Math.sqrt(result.numAttributes()-1));
            }

        }

        /** For each base classifier 
         *      generate random intervals
         *      do the transfrorms
         *      build the classifier
         * */
        while(((System.nanoTime()-t1)+checkpointTimeElapsed < contractTime || trees.size() < numClassifiers)
                && trees.size() < maxClassifiers){

            int i = trees.size();
            //1. Select random intervals for tree i

            int[][] interval =new int[numIntervals][2];  //Start and end

            if (intSelection == 1) {
                if (data.numAttributes() - 1 < minIntervalLength)
                    minIntervalLength = data.numAttributes() - 1;

                for (int j = 0; j < numIntervals; j++) {
                    while (true) {
                        interval[j][0] = rand.nextInt(data.numAttributes() - 1 - minIntervalLength);       //Start point

                        //biased towards min value? (larger effect on small series)

                        //int length = rand.nextInt(data.numAttributes() - 1 - intervals[i][j][0] - minIntervalLength) + minIntervalLength;//Min length 12
                        int length = rand.nextInt(data.numAttributes() - 1 - interval[j][0]);//Min length 3
                        if (length < minIntervalLength)
                            length = minIntervalLength;

                        interval[j][1] = interval[j][0] + length;

                        if (j == 0){
                            break;
                        }

                        int closestIdx = -1;
                        int closestDist = Integer.MAX_VALUE;
                        for (int n = 0; n < j; n++){
                            int dist = Math.abs(interval[n][0] - interval[j][0]);
                            if (dist < closestDist){
                                closestIdx = n;
                                closestDist = dist;
                            }
                        }

                        int overlap = Math.min(interval[j][1], interval[closestIdx][1]) - Math.max(interval[j][0], interval[closestIdx][0]);
                        int closestLength = interval[closestIdx][1] - interval[closestIdx][0];
                        int nonOverlap = closestLength + length - overlap*2;

                        if (nonOverlap >= Math.max(length, closestLength)/2) {
                            break;
                        }
                    }
                }
            }
            else {
                if (data.numAttributes() - 1 < minIntervalLength)
                    minIntervalLength = data.numAttributes() - 1;
                for (int j = 0; j < numIntervals; j++) {
                    interval[j][0] = rand.nextInt(data.numAttributes() - 1 - minIntervalLength);       //Start point
                    int length = rand.nextInt(data.numAttributes() - 1 - interval[j][0]);//Min length 3
                    if (length < minIntervalLength)
                        length = minIntervalLength;
                    interval[j][1] = interval[j][0] + length;
                }
            }

            int[] instInclusions = new int[data.numInstances()];
            if (bagging){
                for (int n = 0; n < data.numInstances(); n++){
                    instInclusions[rand.nextInt(data.numInstances())]++;
                }

                for (int n = 0; n < data.numInstances(); n++){
                    if (instInclusions[n] > 0){
                        inBag[i][n] = true;
                    }
                }
            }
            else{
                for (int n = 0; n < data.numInstances(); n++){
                    instInclusions[n]++;
                }
            }

            if (attSubsample){
                subsampleAtts.add(new ArrayList());

                for (int n = 0; n < ogNumAttributes; n++){
                    subsampleAtts.get(i).add(n);
                }

                while (subsampleAtts.get(i).size() > attSubsampleNum){
                    subsampleAtts.get(i).remove(rand.nextInt(subsampleAtts.get(i).size()));
                }
            }

            int instIdx = 0;
            int lastIdx = -1;
            //2. Generate and store attributes
            for(int k=0;k<data.numInstances();k++){
                //For each instance
                double[] series;
                boolean sameInst = false;
                while (true){
                    if (instInclusions[instIdx] == 0){
                        instIdx++;
                    }
                    else{
                        series = data.instance(instIdx).toDoubleArray();
                        instInclusions[instIdx]--;

                        if (instIdx == lastIdx){
                            result.set(k, new DenseInstance(result.instance(k-1)));
                            sameInst = true;
                        }
                        else{
                            lastIdx = instIdx;
                        }

                        break;
                    }
                }

                if (sameInst) continue;

                if (bagging){
                    result.instance(k).setValue(result.numAttributes()-1,data.instance(instIdx).classValue());
                }

                for(int j=0;j<numIntervals;j++){
                    //extract the interval

//                    FeatureSet f= new FeatureSet();
//                    f.setFeatures(series, intervals[i][j][0], intervals[i][j][1]);
//
//                    result.instance(k).setValue(j*3, f.mean);
//                    result.instance(k).setValue(j*3+1, f.stDev);
//                    result.instance(k).setValue(j*3+2, f.slope);

                    double[] intervalDA = Arrays.copyOfRange(series, interval[j][0], interval[j][1] + 1);

                    if (experimentalOptions == 2 || experimentalOptions == 3) {
                        TSF.FeatureSet f = new TSF.FeatureSet();
                        f.setFeatures(series, interval[j][0], interval[j][1]);
                        result.instance(k).setValue(j * numAttributes + 22, f.mean);
                        result.instance(k).setValue(j * numAttributes + 23, f.stDev);
                        result.instance(k).setValue(j * numAttributes + 24, f.slope);
                    }

                    if (norm) {
                        zNormalise(intervalDA);
                    }

                    if (experimentalOptions > 100) {
                        int offset = 0;

                        for (int g = 0; g < 22; g++) {
                            if (experimentalOptions - 101 == g){
                                offset = 1;
                                continue;
                            }

                            result.instance(k).setValue(j * numAttributes + g - offset, c22.getSummaryStatByIndex(g, k, intervalDA));
                        }
                    }
                    else if (attSubsample){
                        for (int g = 0; g < subsampleAtts.get(i).size(); g++){
                            result.instance(k).setValue(j * numAttributes + g, c22.getSummaryStatByIndex(subsampleAtts.get(i).get(g), k, intervalDA));
                        }
                    }
                    else if (attSubsampleInterval){
                        if (k == 0) {
                            subsampleIntervalAtts[j].add(new ArrayList());

                            for (int n = 0; n < ogNumAttributes; n++) {
                                subsampleIntervalAtts[j].get(i).add(n);
                            }

                            while (subsampleIntervalAtts[j].get(i).size() > attSubsampleNum) {
                                subsampleIntervalAtts[j].get(i).remove(rand.nextInt(subsampleIntervalAtts[j].get(i).size()));
                            }
                        }

                        for (int g = 0; g < subsampleIntervalAtts[j].get(i).size(); g++){
                            result.instance(k).setValue(j * numAttributes + g, c22.getSummaryStatByIndex(subsampleIntervalAtts[j].get(i).get(g), k, intervalDA));
                        }
                    }
                    else{
                        double[] catch22 = c22.transform(intervalDA, -1);

                        for (int g = 0; g < 22; g++) {
                            result.instance(k).setValue(j * numAttributes + g, catch22[g]);
                        }
                    }
                }
            }

            if (bagging){
                result.randomize(rand);
            }

            //3. Create and build tree using all the features. Feature selection
            Classifier tree =AbstractClassifier.makeCopy(base);
            if(seedClassifier && tree instanceof Randomizable)
                ((Randomizable)tree).setSeed(seed*(i+1));

            tree.buildClassifier(result);

            if(bagging && getEstimateOwnPerformance()){
                for(int n=0;n<data.numInstances();n++){
                    if(inBag[i][n])
                        continue;

                    double[] series = data.instance(n).toDoubleArray();
                    for(int j=0;j<numIntervals;j++) {
                        double[] intervalDA = Arrays.copyOfRange(series, interval[j][0], interval[j][1] + 1);
                        double[] catch22 = c22.transform(intervalDA, -1);

                        for (int g = 0; g < 22; g++) {
                            testHolder.instance(0).setValue(j * numAttributes + g, catch22[g]);
                        }
                    }

                    double[] newProbs = tree.distributionForInstance(testHolder.instance(0));
                    oobCounts[n]++;
                    for(int k=0;k<newProbs.length;k++)
                        trainDistributions[n][k]+=newProbs[k];
                }
            }

            trees.add(tree);
            intervals.add(interval);

            if (checkpoint){
                checkpoint(t1);
            }
        }
        long t2=System.nanoTime();

/** Estimate accuracy stage: Three scenarios
 * 1. If we bagged the full build (bagging ==true), we estimate using the full build OOB
 *  If we built on all data (bagging ==false) we estimate either
 *  2. with a 10xCV or (if
 *  3. Build a bagged model simply to get the estimate.
 */
        if(getEstimateOwnPerformance()){
            if(bagging){
                // Use bag data. Normalise probs
                long est1=System.nanoTime();
                double[] preds=new double[data.numInstances()];
                double[] actuals=new double[data.numInstances()];
                long[] predTimes=new long[data.numInstances()];//Dummy variable, need something
                for(int j=0;j<data.numInstances();j++){
                    long predTime = System.nanoTime();
                    for(int k=0;k<trainDistributions[j].length;k++)
                        trainDistributions[j][k]/=oobCounts[j];
                    preds[j]=utilities.GenericTools.indexOfMax(trainDistributions[j]);
                    actuals[j]=data.instance(j).classValue();
                    predTimes[j]=System.nanoTime()-predTime;
                }
                trainResults.addAllPredictions(actuals,preds, trainDistributions, predTimes, null);
                trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
                trainResults.setClassifierName("TSFBagging");
                trainResults.setDatasetName(data.relationName());
                trainResults.setSplit("train");
                trainResults.setFoldID(seed);
                trainResults.setParas(getParameters());
                trainResults.finaliseResults(actuals);
                long est2=System.nanoTime();
                trainResults.setErrorEstimateTime(est2-est1);
            }
//Either do a CV, or bag and get the estimates 
            else if(estimator== EstimatorMethod.CV){
                /** Defaults to 10 or numInstances, whichever is smaller. 
                 * Interface TrainAccuracyEstimate
                 * Could this be handled better? */
                long est1=System.nanoTime();
                int numFolds=setNumberOfFolds(data);
                CrossValidationEvaluator cv = new CrossValidationEvaluator();
                if (seedClassifier)
                    cv.setSeed(seed*5);
                cv.setNumFolds(numFolds);
                Catch22TSF tsf=new Catch22TSF();
                tsf.copyParameters(this);
                if (seedClassifier)
                    tsf.setSeed(seed*100);
                tsf.setEstimateOwnPerformance(false);
                trainResults=cv.evaluate(tsf,data);
                long est2=System.nanoTime();
                trainResults.setErrorEstimateTime(est2-est1);
                trainResults.setClassifierName("TSFCV");
                trainResults.setParas(getParameters());

            }
            else if(estimator== EstimatorMethod.OOB){
                /** Build a single new TSF using Bagging, and extract the estimate from this
                 */
                long est1=System.nanoTime();
                Catch22TSF tsf=new Catch22TSF();
                tsf.copyParameters(this);
                tsf.setSeed(seed);
                tsf.setEstimateOwnPerformance(true);
                tsf.bagging=true;
                tsf.buildClassifier(data);
                trainResults=tsf.trainResults;
                long est2=System.nanoTime();
                trainResults.setErrorEstimateTime(est2-est1);
                trainResults.setClassifierName("TSFOOB");
                trainResults.setParas(getParameters());
            }

            System.out.println("Build time ="+trainResults.getBuildTime());
            if(trainFoldPath!=""){
                trainResults.writeFullResultsToFile(trainFoldPath);
            }
        }
        trainResults.setBuildTime(t2-t1);
    }

    private void copyParameters(Catch22TSF other){
        this.numClassifiers=other.numClassifiers;
        this.numIntervalsFinder=other.numIntervalsFinder;


    }
    public void setEstimatorMethod(String str){
        String s=str.toUpperCase();
        if(s.equals("CV"))
            estimator= EstimatorMethod.CV;
        else if(s.equals("OOB"))
            estimator= EstimatorMethod.OOB;
        else
            throw new UnsupportedOperationException("Unknown estimator methof in TSF = "+str);
    }
    /**
     * @param ins to classifier
     * @return array of doubles: probability of each class
     * @throws Exception
     */
    @Override
    public double[] distributionForInstance(Instance orgIns) throws Exception {
        double[] d=new double[orgIns.numClasses()];

        Instance ins;
        if (preNorm){
            ins = new DenseInstance(orgIns);
            ins.setDataset(orgIns.dataset());
            zNormaliseWithClass(ins);
        }
        else{
            ins = orgIns;
        }

        Catch22 c22 = new Catch22();
        if (outlierNorm) {
            c22.setOutlierNormalise(true);
        }

        //Build transformed instance
        double[] series=ins.toDoubleArray();
        for(int i=0;i<trees.size();i++){
            for(int j=0;j<numIntervals;j++){
                //extract all intervals
//                FeatureSet f= new FeatureSet();
//                f.setFeatures(series, intervals[i][j][0], intervals[i][j][1]);
//                testHolder.instance(0).setValue(j*3, f.mean);
//                testHolder.instance(0).setValue(j*3+1, f.stDev);
//                testHolder.instance(0).setValue(j*3+2, f.slope);

                double[] interval = Arrays.copyOfRange(series, intervals.get(i)[j][0], intervals.get(i)[j][1]+1);

                if (experimentalOptions == 2 || experimentalOptions == 3) {
                    TSF.FeatureSet f = new TSF.FeatureSet();
                    f.setFeatures(series, intervals.get(i)[j][0], intervals.get(i)[j][1]);
                    testHolder.instance(0).setValue(j * numAttributes + 22, f.mean);
                    testHolder.instance(0).setValue(j * numAttributes + 23, f.stDev);
                    testHolder.instance(0).setValue(j * numAttributes + 24, f.slope);
                }

                if (norm){
                    zNormalise(interval);
                }

                if (experimentalOptions > 100) {
                    int offset = 0;

                    for (int g = 0; g < 22; g++) {
                        if (experimentalOptions - 101 == g){
                            offset = 1;
                            continue;
                        }

                        testHolder.instance(0).setValue(j * numAttributes + g - offset, c22.getSummaryStatByIndex(g, j, interval));
                    }
                }
                else if (attSubsample){
                    for (int g = 0; g < subsampleAtts.get(i).size(); g++){
                        testHolder.instance(0).setValue(j * numAttributes + g, c22.getSummaryStatByIndex(subsampleAtts.get(i).get(g), j, interval));
                    }
                }
                else if (attSubsampleInterval){
                    for (int g = 0; g < subsampleIntervalAtts[j].get(i).size(); g++){
                        testHolder.instance(0).setValue(j * numAttributes + g, c22.getSummaryStatByIndex(subsampleIntervalAtts[j].get(i).get(g), j, interval));
                    }
                }
                else {
                    double[] catch22 = c22.transform(interval, -1);

                    for (int g = 0; g < 22; g++) {
                        testHolder.instance(0).setValue(j * numAttributes + g, catch22[g]);
                    }
                }
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
//        System.out.print("TSF para sets ");
//        for (String str:options)
//             System.out.print(","+str);
//        System.out.print("\n");
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

    //
    //
    // TODO: actually implement copy from ser object when finished
    //
    //

    @Override
    public void copyFromSerObject(Object obj) throws Exception {
        if(!(obj instanceof Catch22TSF))
            throw new Exception("The SER file is not an instance of TSF");
        Catch22TSF saved = ((Catch22TSF)obj);
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
            //trainAccuracyEst = saved.trainAccuracyEst;
            //trainCVPath = saved.trainCVPath;
            voteEnsemble = saved.voteEnsemble;
            bagging = saved.bagging;
            inBag = saved.inBag;
            oobCounts = saved.oobCounts;
            trainDistributions = saved.trainDistributions;
            //checkpoint = saved.checkpoint;
            //checkpointPath = saved.checkpointPath
            checkpointTime = saved.checkpointTime;
            checkpointTimeElapsed = saved.checkpointTime; //intentional, time spent building previously unchanged
            //checkpointTimeElapsed -= System.nanoTime()-t1; //look at cboss
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

    //
    //
    // TODO: actually implement checkpoitning when finished
    //
    //

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

////Nested class to store three simple summary features used to construct train data
//    public static class FeatureSet{
//        public static boolean findSkew=false;
//        public static boolean findKurtosis=false;
//        double mean;
//        double stDev;
//        double slope;
//        double skew;
//        double kurtosis;
//        public void setFeatures(double[] data, int start, int end){
//            double sumX=0,sumYY=0;
//            double sumY3=0,sumY4=0;
//            double sumY=0,sumXY=0,sumXX=0;
//            int length=end-start+1;
//            for(int i=start;i<=end;i++){
//                sumY+=data[i];
//                sumYY+=data[i]*data[i];
//                sumX+=(i-start);
//                sumXX+=(i-start)*(i-start);
//                sumXY+=data[i]*(i-start);
//            }
//            mean=sumY/length;
//            stDev=sumYY-(sumY*sumY)/length;
//            slope=(sumXY-(sumX*sumY)/length);
//            double denom=sumXX-(sumX*sumX)/length;
//            if(denom!=0)
//                slope/=denom;
//            else
//                slope=0;
//            stDev/=length;
//            if(stDev==0)    //Flat line
//                slope=0;
////            else //Why not doing this? Because not needed?
////                stDev=Math.sqrt(stDev);
//            if(slope==0)
//                stDev=0;
//            if(findSkew){
//                if(stDev==0)
//                    skew=1;
//                else{
//                    for(int i=start;i<=end;i++)
//                        sumY3+=data[i]*data[i]*data[i];
//                    skew=sumY3-3*sumY*sumYY+2*sumY*sumY;
//                    skew/=length*stDev*stDev*stDev;
//                }
//            }
//            if(findKurtosis){
//                if(stDev==0)
//                    kurtosis=1;
//                else{
//                    for(int i=start;i<=end;i++)
//                        sumY4+=data[i]*data[i]*data[i]*data[i];
//                    kurtosis=sumY4-4*sumY*sumY3+6*sumY*sumY*sumYY-3*sumY*sumY*sumY*sumY;
//                    skew/=length*stDev*stDev*stDev*stDev;
//                }
//            }
//
//        }
//        public void setFeatures(double[] data){
//            setFeatures(data,0,data.length-1);
//        }
//        @Override
//        public String toString(){
//            return "mean="+mean+" stdev = "+stDev+" slope ="+slope;
//        }
//    }

    public static void main(String[] arg) throws Exception{

//        System.out.println(ClassifierTools.testUtils_getIPDAcc(new TSF(0)));
//        System.out.println(ClassifierTools.testUtils_confirmIPDReproduction(new TSF(0), 0.967930029154519, "2019/09/25"));

// Basic correctness tests, including setting paras through 
        String dataLocation="Z:\\ArchiveData\\Univariate_arff\\";
//        String resultsLocation="C:\\temp\\";
        String problem="ItalyPowerDemand";
//        File f= new File(resultsLocation+problem);
//        if(!f.isDirectory())
//            f.mkdirs();
        Instances train=DatasetLoading.loadDataNullable(dataLocation+problem+"\\"+problem+"_TRAIN");
        Instances test=DatasetLoading.loadDataNullable(dataLocation+problem+"\\"+problem+"_TEST");
        Catch22TSF tsf = new Catch22TSF();
        tsf.setSeed(0);
        tsf.outlierNorm = true;
        tsf.attSubsample = true;
        //tsf.setTrainTimeLimit(TimeUnit.SECONDS,5);
//        tsf.writeTrainEstimatesToFile(resultsLocation+problem+"trainFold0.csv");
        double a;
        long t1 = System.nanoTime();
        tsf.buildClassifier(train);
        System.out.println("Train time="+(System.nanoTime()-t1)*1e-9);
        System.out.println("build ok: original atts="+(train.numAttributes()-1)+" new atts ="+(tsf.testHolder.numAttributes()-1)+" num trees = "+tsf.trees.size()+" num intervals = "+tsf.numIntervals);
        a=ClassifierTools.accuracy(test, tsf);
        System.out.println("Test Accuracy ="+a);
        String[] options=new String[4];
        options[0]="-T";
        options[1]="100";
        options[2]="-I";
        options[3]="1";
        tsf.setOptions(options);
        //tsf.setTrainTimeLimit(TimeUnit.SECONDS,2);
        t1 = System.nanoTime();
        tsf.buildClassifier(train);
        System.out.println("Train time="+(System.nanoTime()-t1)*1e-9);
        System.out.println("build ok: original atts="+(train.numAttributes()-1)+" new atts ="+(tsf.testHolder.numAttributes()-1)+" num trees = "+tsf.trees.size()+" num intervals = "+tsf.numIntervals);
        a=ClassifierTools.accuracy(test, tsf);
        System.out.println("Test Accuracy ="+a);


    }

    @Override
    public ParameterSpace getDefaultParameterSearchSpace(){
        //TUNED TSC Classifiers
        /* Valid options are: <p/>
         * <pre> -T Number of trees.</pre>
         * <pre> -I Number of intervals to fit.</pre>
         */
        ParameterSpace ps=new ParameterSpace();
        String[] numTrees={"100","200","300","400","500","600","700","800","900","1000"};
        ps.addParameter("-T", numTrees);
        String[] numInterv={"sqrt","log","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"};
        ps.addParameter("-I", numInterv);
        return ps;
    }


}
  