
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
import experiments.data.DatasetLoading;
import machine_learning.classifiers.ContinuousIntervalTree;
import tsml.classifiers.*;
import tsml.transformers.ColumnNormalizer;
import tsml.transformers.Differences;
import tsml.transformers.PowerSpectrum;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.TimeUnit;

import static utilities.ArrayUtilities.sum;
import static utilities.StatisticalUtilities.median;
import static utilities.Utilities.extractTimeSeries;

/**
 * Implementation of the Supervised Time Series Forest
 * This classifier is Contractable, Checkpointable and can estimate performance from the train data internally.
 *
 * @author Matthew Middlehurst
*/
public class STSF extends EnhancedAbstractClassifier implements TechnicalInformationHandler, TrainTimeContractable {
    //Static defaults
    private final static int DEFAULT_NUM_CLASSIFIERS=500;

    /** Primary parameters potentially tunable*/
    private int numClassifiers=DEFAULT_NUM_CLASSIFIERS;

    /** Ensemble members of base classifier, default to random forest RandomTree */
    private ArrayList<Classifier> trees;
    private Classifier classifier = new ContinuousIntervalTree();

    /** for each classifier i representation r attribute a interval j  starts at intervals[i][r][a][j][0] and
     ends  at  intervals[i][r][a][j][1] */
    private ArrayList<ArrayList<int[]>[][]> intervals;

    /**Holding variable for test classification in order to retain the header info*/
    private ArrayList<Instances> testHolders;

    /** voteEnsemble determines whether to aggregate classifications or
     * probabilities when predicting */
    private boolean voteEnsemble=true;

    private int numInstances;
    private int newNumInstances;

    private boolean trainTimeContract = false;
    transient private long trainContractTimeNanos = 0;
    transient private long finalBuildtrainContractTimeNanos = 0;

    private PowerSpectrum ps = new PowerSpectrum();
    private Differences di = new Differences();

    protected static final long serialVersionUID = 32554L;

    public STSF(){
        //STSF Has the capability to form train estimates
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
    }
    public STSF(int s){
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
        setSeed(s);
    }

    /**
     *
     * @param c a base classifier constructed elsewhere and cloned into ensemble
     */
    public void setBaseClassifier(Classifier c){
        classifier =c;
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
        int numTrees = trees == null ? 0 : trees.size();
        String result=super.getParameters()+",numTrees,"+numTrees+",voting,"+voteEnsemble+",BaseClassifier,"+
                classifier.getClass().getSimpleName();

        if(trainTimeContract)
            result+= ",trainContractTimeNanos," +trainContractTimeNanos;
        else
            result+=",NoContract";
//Any other contract information here

        result+=",EstimateOwnPerformance,"+getEstimateOwnPerformance();
        if(getEstimateOwnPerformance())
            result+=",EstimateMethod,"+estimator;
        return result;
 
    }

    public void setNumTrees(int t){
        numClassifiers=t;
    }

 /**
  * paper defining STSF
  * @return TechnicalInformation
  */
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "C. Nestor, N. Elham, Q. Jianzhong and K. Lars");
        result.setValue(TechnicalInformation.Field.YEAR, "2020");
        result.setValue(TechnicalInformation.Field.TITLE, "Fast and Accurate Time Series Classification Through Supervised Interval Search");
        result.setValue(TechnicalInformation.Field.JOURNAL, "IEEE International Conference on Data Mining");
        return result;
    }

    /**
     * main buildClassifier
     * @param data
     * @throws Exception
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        super.buildClassifier(data);

        // can classifier handle the data?
        getCapabilities().testWithFail(data);
        long startTime=System.nanoTime();

        numInstances = data.numInstances();

        trees = new ArrayList(numClassifiers);
        intervals = new ArrayList();
        testHolders = new ArrayList();

        finalBuildtrainContractTimeNanos=trainContractTimeNanos;
        //If contracted and estimating own performance, distribute the contract evenly between estimation and the final build
        if(trainTimeContract && getEstimateOwnPerformance()){
            finalBuildtrainContractTimeNanos/=2;
            printLineDebug(" Setting final contract time to "+finalBuildtrainContractTimeNanos+" nanos");
        }

        Instances[] representations = new Instances[3];
        representations[0] = new Instances(data);

        ArrayList<Integer>[] idxByClass = new ArrayList[data.numClasses()];
        for (int i = 0; i < idxByClass.length; i++){
            idxByClass[i] = new ArrayList<>();
        }
        for (int i = 0; i < data.numInstances(); i++){
            idxByClass[(int)data.get(i).classValue()].add(i);
        }

        double average = (double)data.numInstances()/data.numClasses();
        int[] instToAdd = new int[numInstances];
        for (int i = 0; i < idxByClass.length; i++) {
            if (idxByClass[i].size() < average) {
                int n = idxByClass[i].size();
                while (n < average) {
                    instToAdd[idxByClass[i].get(rand.nextInt(idxByClass[i].size()))]++;
                    n++;
                }
            }
        }

        newNumInstances = numInstances + sum(instToAdd);

        ps = new PowerSpectrum();
        representations[1] = ps.transform(representations[0]);
        di = new Differences();
        di.setSubtractFormerValue(true);
        representations[2] = di.transform(representations[0]);

        int classifiersBuilt = trees.size();

        /** MAIN BUILD LOOP
         *  For each base classifier
         *      generate random intervals
         *      do the transforms
         *      build the classifier
         * */
        while(withinTrainContract(startTime) && (classifiersBuilt < numClassifiers)) {
            if (classifiersBuilt % 100 == 0)
                printLineDebug("\t\t\t\t\tBuilding STSF tree " + classifiersBuilt + " time taken = " + (System.nanoTime() - startTime) + " contract =" + finalBuildtrainContractTimeNanos + " nanos");

            //If bagging find instances with replacement
            int[] instInclusions = new int[numInstances];
            int[] baggingClassCounts = new int[numClasses];;
            for (int n = 0; n < numInstances; n++) {
                instInclusions[rand.nextInt(numInstances)]++;
                instInclusions[n] += instToAdd[n];
            }

            for (int n = 0; n < numInstances; n++) {
                if (instInclusions[n] > 0) {
                    baggingClassCounts[(int)representations[0].get(n).classValue()] += instInclusions[n];
                }
            }

            //1. Select intervals for tree i
            intervals.add(new ArrayList[3][]);
            int totalAtts = 0;
            for (int r = 0; r < representations.length; r++) {
                intervals.get(classifiersBuilt)[r] = findCandidateDiscriminatoryIntervals(representations[r],
                        instInclusions, baggingClassCounts);

                for (int a = 0; a < intervals.get(classifiersBuilt)[r].length; a++) {
                    totalAtts += intervals.get(classifiersBuilt)[r][a].size();
                }
            }

            //2. Generate and store attributes
            ArrayList<Attribute> atts = new ArrayList<>();
            for (int j = 0; j < totalAtts; j++) {
                atts.add(new Attribute("att" + j));
            }
            atts.add(data.classAttribute());
            //create blank instances with the correct class value
            Instances result = new Instances("Tree", atts, newNumInstances);
            result.setClassIndex(result.numAttributes() - 1);

            Instances testHolder = new Instances(result, 0);
            testHolder.add(new DenseInstance(result.numAttributes()));
            testHolders.add(testHolder);

            //For bagging
            int instIdx = 0;
            int lastIdx = -1;

            for (int n = 0; n < newNumInstances; n++) {
                boolean sameInst = false;

                while (true) {
                    if (instInclusions[instIdx] == 0) {
                        instIdx++;
                    } else {
                        instInclusions[instIdx]--;

                        if (instIdx == lastIdx) {
                            result.add(n, new DenseInstance(result.instance(n - 1)));
                            sameInst = true;
                        } else {
                            lastIdx = instIdx;
                        }

                        break;
                    }
                }

                if (sameInst) continue;

                DenseInstance in = new DenseInstance(result.numAttributes());
                in.setValue(result.numAttributes() - 1, representations[0].instance(instIdx).classValue());

                int p = 0;
                for (int r = 0; r < representations.length; r++) {
                    double[] series = extractTimeSeries(representations[r].get(instIdx));

                    for (int a = 0; a < FeatureSet.numFeatures; a++) {
                        for (int j = 0; j < intervals.get(classifiersBuilt)[r][a].size(); j++) {
                            int[] interval = intervals.get(classifiersBuilt)[r][a].get(j);
                            double val = FeatureSet.calcFeatureByIndex(a, interval[0], interval[1], series);
                            in.setValue(p, val);
                            p++;
                        }
                    }
                }

                result.add(in);
            }

            //3. Create and build tree using all the features.
            Classifier tree = AbstractClassifier.makeCopy(classifier);
            if (seedClassifier && tree instanceof Randomizable)
                ((Randomizable) tree).setSeed(seed * (classifiersBuilt + 1));

            tree.buildClassifier(result);
            trees.add(tree);
            classifiersBuilt++;
        }

        if(classifiersBuilt==0){//Not enough time to build a single classifier
            throw new Exception((" ERROR in STSF, no trees built, contract time probably too low. Contract time ="+trainContractTimeNanos));
        }

        long endTime=System.nanoTime();
        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        trainResults.setBuildTime(endTime-startTime-trainResults.getErrorEstimateTime());
        trainResults.setBuildPlusEstimateTime(trainResults.getBuildTime());
        /** Estimate accuracy from Train data
         * distributions and predictions stored in trainResults */
        if(getEstimateOwnPerformance()){
            long est1=System.nanoTime();
            estimateOwnPerformance(data);
            long est2=System.nanoTime();
            trainResults.setErrorEstimateTime(est2-est1);
            trainResults.setBuildPlusEstimateTime(trainResults.getBuildTime()+trainResults.getErrorEstimateTime());
        }
        trainResults.setParas(getParameters());
        printLineDebug("*************** Finished STSF Build with "+classifiersBuilt+" Trees built in "+(System.nanoTime()-startTime)/1000000000+" Seconds  ***************");
    }

    private ArrayList<int[]>[] findCandidateDiscriminatoryIntervals(Instances rep, int[] instInclusions,
                                                                    int[] classCounts){
        int seriesLength = rep.numAttributes()-1;
        int splitPoint;
        if (seriesLength <= 8) splitPoint = seriesLength/2;
        else splitPoint = rand.nextInt(seriesLength-8)+4; //min 4, max serieslength-4

        ColumnNormalizer rn = new ColumnNormalizer();
        rn.fit(rep);
        rn.setNormMethod(ColumnNormalizer.NormType.STD_NORMAL);
        Instances data = rn.transform(rep);

        ArrayList<int[]>[] newIntervals = new ArrayList[FeatureSet.numFeatures];
        for (int i = 0; i < FeatureSet.numFeatures; i++){
            newIntervals[i] = new ArrayList<>();
            supervisedIntervalSearch(data, instInclusions, i, newIntervals[i], classCounts, 0, splitPoint);
            supervisedIntervalSearch(data, instInclusions, i, newIntervals[i], classCounts, splitPoint+1,
                    rep.numAttributes()-2);
        }

        return newIntervals;
    }

    private void supervisedIntervalSearch(Instances data, int[] instInclusions, int featureIdx,
                                          ArrayList<int[]> intervals, int[] classCount, int start, int end){
        int seriesLength = end-start;
        if (seriesLength < 4) return;
        int halfSeriesLength = seriesLength/2;

        double[] x1 = new double[newNumInstances];
        double[] x2 = new double[newNumInstances];
        double[] y = new double[newNumInstances];

        int e1 = start + halfSeriesLength;
        int e2 = start + halfSeriesLength + 1;
        int instIdx = 0;
        int lastIdx = -1;
        int[] instInclusionsCopy = Arrays.copyOf(instInclusions, instInclusions.length);

        for (int i = 0; i < newNumInstances; i++){
            boolean sameInst = false;

            while (true) {
                if (instInclusionsCopy[instIdx] == 0) {
                    instIdx++;
                } else {
                    instInclusionsCopy[instIdx]--;

                    if (instIdx == lastIdx) {
                        x1[i] = x1[i-1];
                        x1[i] = x1[i-1];
                        y[i] = y[i-1];
                        sameInst = true;
                    } else {
                        lastIdx = instIdx;
                    }

                    break;
                }
            }

            if (sameInst) continue;

            double[] series = data.instance(instIdx).toDoubleArray();
            x1[i] = FeatureSet.calcFeatureByIndex(featureIdx, start, e1, series);
            x2[i] = FeatureSet.calcFeatureByIndex(featureIdx, e2, end, series);
            y[i] = series[series.length-1];
        }

        double s1 = fisherScore(x1, y, classCount);
        double s2 = fisherScore(x2, y, classCount);

        if (s2 < s1){
            intervals.add(new int[]{start, e1});
            supervisedIntervalSearch(data, instInclusions, featureIdx, intervals, classCount, start, e1);
        }
        else{
            intervals.add(new int[]{e2, end});
            supervisedIntervalSearch(data, instInclusions, featureIdx, intervals, classCount, e2, end);
        }
    }

    private double fisherScore(double[] x, double[] y, int[] classCounts){
        double a = 0, b = 0;

        double xMean = 0;
        for (int n = 0; n < x.length; n++){
            xMean += x[n];
        }
        xMean /= x.length;

        for (int i = 0; i < classCounts.length; i++){
            double xyMean = 0;
            for (int n = 0; n < x.length; n++){
                if (i == y[n]) {
                    xyMean += x[n];
                }
            }
            xyMean /= classCounts[i];

            double squareSum = 0;
            for (int n = 0; n < x.length; n++){
                if (i == y[n]) {
                    double temp = x[n] - xyMean;
                    squareSum += temp * temp;
                }
            }
            double xyStdev = classCounts[i]-1 == 0 ? 0 : Math.sqrt(squareSum/(classCounts[i]-1));

            a += classCounts[i]*Math.pow(xyMean-xMean, 2);
            b += classCounts[i]*Math.pow(xyStdev, 2);
        }

        return b == 0 ? 0 : a/b;
    }

    /**
     * estimating own performance
     *  Three scenarios
     *          1. If we bagged the full build (bagging ==true), we estimate using the full build OOB. Assumes the final
     *          model has already been built
     *           If we built on all data (bagging ==false) we estimate either
     *              2. with a 10xCV if estimator==EstimatorMethod.CV
     *              3. Build a bagged model simply to get the estimate estimator==EstimatorMethod.OOB
     *    Note that all this needs to come out of any contract time we specify.
     * @param data
     * @throws Exception from distributionForInstance
     */
    private void estimateOwnPerformance(Instances data) throws Exception {
        // Defaults to 10 or numInstances, whichever is smaller.
        int numFolds=setNumberOfFolds(data);
        CrossValidationEvaluator cv = new CrossValidationEvaluator();
        if (seedClassifier)
            cv.setSeed(seed*5);
        cv.setNumFolds(numFolds);
        STSF tsf=new STSF();
        tsf.copyParameters(this);
        tsf.setDebug(this.debug);
        if (seedClassifier)
            tsf.setSeed(seed*100);
        tsf.setEstimateOwnPerformance(false);
        if(trainTimeContract)//Need to split the contract time, will give time/(numFolds+2) to each fio
            tsf.setTrainTimeLimit(finalBuildtrainContractTimeNanos/numFolds);
        printLineDebug(" Doing CV evaluation estimate performance with  "+tsf.getTrainContractTimeNanos()/1000000000+" secs per fold.");
        long buildTime = trainResults.getBuildTime();
        trainResults=cv.evaluate(tsf,data);
        trainResults.setBuildTime(buildTime);
        trainResults.setClassifierName("TSFCV");
        trainResults.setErrorEstimateMethod("CV_"+numFolds);
    }
     
    private void copyParameters(STSF other){
        this.numClassifiers=other.numClassifiers;
    }

    @Override
    public long getTrainContractTimeNanos(){
            return trainContractTimeNanos;
    }

    /**
     * @param ins to classifier
     * @return array of doubles: probability of each class
     * @throws Exception
     */
    @Override
    public double[] distributionForInstance(Instance ins) throws Exception {
        double[] d=new double[ins.numClasses()];
        //Build transformed instance

        double[][] representations = new double[3][];
        representations[0] = ins.toDoubleArray();
        representations[1] = ps.transform(ins).toDoubleArray();
        representations[2] = di.transform(ins).toDoubleArray();

        for(int i=0;i<trees.size();i++){
            Instances testHolder = testHolders.get(i);

            int p = 0;
            for (int r = 0; r < representations.length; r++){
                for (int a = 0; a < FeatureSet.numFeatures; a++){
                    for (int j = 0; j < intervals.get(i)[r][a].size(); j++){
                        int[] interval = intervals.get(i)[r][a].get(j);
                        double val = FeatureSet.calcFeatureByIndex(a, interval[0], interval[1], representations[r]);
                        testHolder.instance(0).setValue(p, val);
                        p++;
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
        if(sum>0)
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
        return findIndexOfMax(d, rand);
    }

    @Override//TrainTimeContractable
    public void setTrainTimeLimit(long amount) {
        printLineDebug("STSF setting contract to "+amount);

        if(amount>0) {
            trainContractTimeNanos = amount;
            trainTimeContract = true;
        }
        else
            trainTimeContract = false;
    }

    @Override//TrainTimeContractable
    public boolean withinTrainContract(long start){
        if(trainContractTimeNanos<=0) return true; //Not contracted
        return System.nanoTime()-start < finalBuildtrainContractTimeNanos;
    }
 
    //Nested class to store seven simple summary features used to construct train data
    private static class FeatureSet{
        static int numFeatures = 7;

        public static double calcFeatureByIndex(int idx, int start, int end, double[] data) {
            switch (idx){
                case 0: return calcMean(start, end, data);
                case 1: return calcMedian(start, end, data);
                case 2: return calcStandardDeviation(start, end, data);
                case 3: return calcSlope(start, end, data);
                case 4: return calcInterquartileRange(start, end, data);
                case 5: return calcMin(start, end, data);
                case 6: return calcMax(start, end, data);
                default: return Double.NaN;
            }
        }

        public static double calcMean(int start, int end, double[] data){
            double sumY = 0;
            for(int i=start;i<=end;i++) {
                sumY += data[i];
            }

            int length = end-start+1;
            return sumY/length;
        }

        public static double calcMedian(int start, int end, double[] data){
            ArrayList<Double> sortedData = new ArrayList<>(end-start+1);
            for(int i=start;i<=end;i++){
                sortedData.add(data[i]);
            }

            return median(sortedData, false); //sorted in function
        }

        public static double calcStandardDeviation(int start, int end, double[] data){
            double sumY = 0;
            double sumYY = 0;
            for(int i=start;i<=end;i++) {
                sumY += data[i];
                sumYY += data[i] * data[i];
            }

            int length = (end-start)+1;
            return (sumYY-(sumY*sumY)/length)/(length-1);
        }

        public static double calcSlope(int start, int end, double[] data){
            double sumY = 0;
            double sumX = 0, sumXX = 0, sumXY = 0;
            for(int i=start;i<=end;i++) {
                sumY += data[i];
                sumX+=(i-start);
                sumXX+=(i-start)*(i-start);
                sumXY+=data[i]*(i-start);
            }

            int length = end-start+1;
            double slope=(sumXY-(sumX*sumY)/length);
            double denom=sumXX-(sumX*sumX)/length;
            slope = denom == 0 ? 0 : slope/denom;
            return slope;
        }

        public static double calcInterquartileRange(int start, int end, double[] data){
            ArrayList<Double> sortedData = new ArrayList<>(end-start+1);
            for(int i=start;i<=end;i++){
                sortedData.add(data[i]);
            }
            Collections.sort(sortedData);

            int length = end-start+1;
            ArrayList<Double> left = new ArrayList<>(length / 2 + 1);
            ArrayList<Double> right = new ArrayList<>(length / 2 + 1);
            if (length % 2 == 1) {
                for (int i = 0; i <= length / 2; i++){
                    left.add(sortedData.get(i));
                }
            }
            else {
                for (int i = 0; i < length / 2; i++){
                    left.add(sortedData.get(i));
                }

            }
            for (int i = length / 2; i < sortedData.size(); i++){
                right.add(sortedData.get(i));
            }

            return median(right, false) - median(left, false);
        }

        public static double calcMin(int start, int end, double[] data){
            double min = Double.MAX_VALUE;
            for(int i=start;i<=end;i++){
                if (data[i] < min) min = data[i];
            }
            return min;
        }

        public static double calcMax(int start, int end, double[] data){
            double max = -999999999;
            for(int i=start;i<=end;i++){
                if (data[i] > max) max = data[i];
            }
            return max;
        }
    }

    public static void main(String[] arg) throws Exception{
        // Basic correctness tests, including setting paras through
        String dataLocation="Z:\\ArchiveData\\Univariate_arff\\";
        String problem="ItalyPowerDemand";
        Instances train=DatasetLoading.loadDataNullable(dataLocation+problem+"\\"+problem+"_TRAIN");
        Instances test=DatasetLoading.loadDataNullable(dataLocation+problem+"\\"+problem+"_TEST");
        STSF tsf = new STSF();
        tsf.setSeed(0);
        double a;
        tsf.buildClassifier(train);
        System.out.println(tsf.trainResults.getBuildTime());
        a=ClassifierTools.accuracy(test, tsf);
        System.out.println("Test Accuracy ="+a);
        System.out.println();
    }
}
  