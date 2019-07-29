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
package timeseriesweka.classifiers.dictionary_based;

import java.security.InvalidParameterException;
import java.util.*;

import net.sourceforge.sizeof.SizeOf;
import timeseriesweka.classifiers.*;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import utilities.*;
import utilities.samplers.*;
import weka.classifiers.functions.GaussianProcesses;
import weka.core.*;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;

import static utilities.InstanceTools.resampleTrainAndTestInstances;
import static utilities.multivariate_tools.MultivariateInstanceTools.*;
import static weka.core.Utils.sum;

/**
 * BOSS classifier with parameter search and ensembling for univariate and
 * multivariate time series classification.
 * If parameters are known, use the nested class BOSSIndividual and directly provide them.
 *
 * Options to change the method of ensembling to randomly select parameters instead of searching.
 * Has the capability to contract train time and checkpoint when using a random ensemble.
 *
 * Alphabetsize fixed to four and maximum wordLength of 16.
 *
 * @author James Large, updated by Matthew Middlehurst
 *
 * Implementation based on the algorithm described in getTechnicalInformation()
 */
public class BOSS extends AbstractClassifierWithTrainingInfo implements TrainAccuracyEstimator,
        TechnicalInformationHandler, MultiThreadable {

    private int seed = 0;
    private Random rand;

    private transient LinkedList<BOSSIndividual>[] classifiers;
    private int numSeries;
    private int[] numClassifiers;
    private int currentSeries = 0;
    private boolean isMultivariate = false;

    private final int[] wordLengths = { 16, 14, 12, 10, 8 };
    private final int[] alphabetSize = { 4 };
    private final boolean[] normOptions = { true, false };
    private final double correctThreshold = 0.92;
    private int maxEnsembleSize = 500;

    private String trainCVPath;
    private boolean trainCV = false;

    private transient Instances train;
    private double ensembleCvAcc = -1;
    private double[] ensembleCvPreds = null;

    private int numThreads = 1;
    private boolean multiThread = false;
    private ExecutorService ex;

    protected static final long serialVersionUID = 22554L;

    public BOSS() {}

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "P. Schafer");
        result.setValue(TechnicalInformation.Field.TITLE, "The BOSS is concerned with time series classification in the presence of noise");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Data Mining and Knowledge Discovery");
        result.setValue(TechnicalInformation.Field.VOLUME, "29");
        result.setValue(TechnicalInformation.Field.NUMBER,"6");
        result.setValue(TechnicalInformation.Field.PAGES, "1505-1530");
        result.setValue(TechnicalInformation.Field.YEAR, "2015");
        return result;
    }

    @Override
    public Capabilities getCapabilities(){
        Capabilities result = super.getCapabilities();
        result.disableAll();

        result.setMinimumNumberInstances(2);

        // attributes
        result.enable(Capabilities.Capability.RELATIONAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        return result;
    }

    @Override
    public String getParameters() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.getParameters());

        sb.append(",numSeries,").append(numSeries);

        for (int n = 0; n < numSeries; n++) {
            sb.append(",numclassifiers,").append(n).append(",").append(numClassifiers[n]);

            for (int i = 0; i < numClassifiers[n]; ++i) {
                BOSSIndividual boss = classifiers[n].get(i);
                sb.append(",windowSize,").append(boss.getWindowSize()).append(",wordLength,").append(boss.getWordLength());
                sb.append(",alphabetSize,").append(boss.getAlphabetSize()).append(",norm,").append(boss.isNorm());
            }
        }

        return sb.toString();
    }

    @Override
    public void setThreadAllowance(int numThreads) {
        if (numThreads > 1) {
            this.numThreads = numThreads;
            multiThread = true;
        }
        else{
            this.numThreads = 1;
            multiThread = false;
        }
    }

    @Override
    public void writeTrainEstimatesToFile(String outputPathAndName){
        trainCVPath = outputPathAndName;
        trainCV = true;
    }

    @Override
    public void setFindTrainAccuracyEstimate(boolean setCV){
        trainCV = setCV;
    }

    @Override
    public boolean findsTrainAccuracyEstimate(){ return trainCV; }

    @Override
    public ClassifierResults getTrainResults(){
        trainResults.setAcc(ensembleCvAcc);
        return trainResults;
    }

    public void setMaxEnsembleSize(int size) {
        maxEnsembleSize = size;
    }

    public void setSeed(int i) {
        seed = i;
    }

    @Override
    public void buildClassifier(final Instances data) throws Exception {
        trainResults.setBuildTime(System.nanoTime());

        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        if(data.checkForAttributeType(Attribute.RELATIONAL)){
            isMultivariate = true;
        }

        //Window length settings
        int seriesLength = isMultivariate ? channelLength(data)-1 : data.numAttributes()-1; //minus class attribute
        int minWindow = 10;
        double maxWinLenProportion = 1;
        int maxWindow = (int)(seriesLength* maxWinLenProportion);
        if (maxWindow < minWindow) minWindow = maxWindow/2;
        //whats the max number of window sizes that should be searched through
        double maxWinSearchProportion = 0.25;
        double maxWindowSearches = seriesLength* maxWinSearchProportion;
        int winInc = (int)((maxWindow - minWindow) / maxWindowSearches);
        if (winInc < 1) winInc = 1;

        //initialise variables
        if (data.classIndex() != data.numAttributes()-1)
            throw new Exception("BOSS_BuildClassifier: Class attribute not set as last attribute in dataset");

        //Multivariate
        if (isMultivariate) {
            numSeries = numChannels(data);
            classifiers = new LinkedList[numSeries];

            for (int n = 0; n < numSeries; n++){
                classifiers[n] = new LinkedList<>();
            }

            numClassifiers = new int[numSeries];
        }
        //Univariate
        else{
            numSeries = 1;
            classifiers = new LinkedList[1];
            classifiers[0] = new LinkedList<>();
            numClassifiers = new int[1];
        }

        rand = new Random(seed);

        this.train = data;

        if (multiThread && numThreads == 1){
            numThreads = Runtime.getRuntime().availableProcessors();
        }

        //required to deal with multivariate datasets, each channel is split into its own instances
        Instances[] series;

        //Multivariate
        if (isMultivariate) {
            series = splitMultivariateInstances(data);
        }
        //Univariate
        else{
            series = new Instances[1];
            series[0] = data;
        }

        for (int n = 0; n < numSeries; n++) {
            currentSeries = n;
            double maxAcc = -1.0;

            //the acc of the worst member to make it into the final ensemble as it stands
            double minMaxAcc = -1.0;

            for (boolean normalise : normOptions) {
                for (int winSize = minWindow; winSize <= maxWindow; winSize += winInc) {
                    BOSSIndividual boss = new BOSSIndividual(wordLengths[0], alphabetSize[0], winSize, normalise, multiThread, numThreads, ex);
                    boss.seed = seed;
                    boss.buildClassifier(series[n]); //initial setup for this windowsize, with max word length

                    BOSSIndividual bestClassifierForWinSize = null;
                    double bestAccForWinSize = -1.0;

                    //find best word length for this window size
                    for (Integer wordLen : wordLengths) {
                        boss = boss.buildShortenedBags(wordLen); //in first iteration, same lengths (wordLengths[0]), will do nothing

                        double acc = individualTrainAcc(boss, series[n], bestAccForWinSize);

                        if (acc >= bestAccForWinSize) {
                            bestAccForWinSize = acc;
                            bestClassifierForWinSize = boss;
                        }
                    }

                    //if this window size's accuracy is not good enough to make it into the ensemble, dont bother storing at all
                    if (makesItIntoEnsemble(bestAccForWinSize, maxAcc, minMaxAcc, classifiers[n].size())) {
                        bestClassifierForWinSize.clean();
                        bestClassifierForWinSize.accuracy = bestAccForWinSize;
                        classifiers[n].add(bestClassifierForWinSize);

                        if (bestAccForWinSize > maxAcc) {
                            maxAcc = bestAccForWinSize;
                            //get rid of any extras that dont fall within the new max threshold
                            Iterator<BOSSIndividual> it = classifiers[n].iterator();
                            while (it.hasNext()) {
                                BOSSIndividual b = it.next();
                                if (b.accuracy < maxAcc * correctThreshold) {
                                    it.remove();
                                }
                            }
                        }

                        while (classifiers[n].size() > maxEnsembleSize) {
                            //cull the 'worst of the best' until back under the max size
                            int minAccInd = (int) findMinEnsembleAcc()[0];

                            classifiers[n].remove(minAccInd);
                        }

                        minMaxAcc = findMinEnsembleAcc()[1]; //new 'worst of the best' acc
                    }

                    numClassifiers[n] = classifiers[n].size();
                }
            }
        }

        //end train time in nanoseconds
        trainResults.setBuildTime(System.nanoTime() - trainResults.getBuildTime());

        //Estimate train accuracy
        if (trainCV) {
            trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
            trainResults.setClassifierName("BOSS");
            trainResults.setDatasetName(data.relationName());
            trainResults.setFoldID(seed);
            trainResults.setSplit("train");
            trainResults.setParas(getParameters());
            double result = findEnsembleTrainAcc(data);
            trainResults.finaliseResults();
            trainResults.writeFullResultsToFile(trainCVPath);

            System.out.println("CV acc ="+result);

            trainCV = false;
        }
    }

    //[0] = index, [1] = acc
    private double[] findMinEnsembleAcc() {
        double minAcc = Double.MAX_VALUE;
        int minAccInd = 0;
        for (int i = 0; i < classifiers[currentSeries].size(); ++i) {
            double curacc = classifiers[currentSeries].get(i).accuracy;
            if (curacc < minAcc) {
                minAcc = curacc;
                minAccInd = i;
            }
        }

        return new double[] { minAccInd, minAcc };
    }

    private double individualTrainAcc(BOSSIndividual boss, Instances series, double lowestAcc) throws Exception {
        int correct = 0;
        int numInst = series.numInstances();
        int requiredCorrect = (int)(lowestAcc*numInst);

        if (multiThread){
            ex = Executors.newFixedThreadPool(numThreads);
            ArrayList<BOSSIndividual.TrainNearestNeighbourThread> threads = new ArrayList<>(sum(numClassifiers));

            for (int i = 0; i < numInst; ++i) {
                BOSSIndividual.TrainNearestNeighbourThread t = boss.new TrainNearestNeighbourThread(i);
                threads.add(t);
                ex.execute(t);
            }

            ex.shutdown();
            while (!ex.isTerminated());

            for (BOSSIndividual.TrainNearestNeighbourThread t: threads){
                if (t.nn == series.get(t.testIndex).classValue()) {
                    ++correct;
                }
            }
        }
        else {
            for (int i = 0; i < numInst; ++i) {
                if (correct + numInst - i < requiredCorrect) {
                    return -1;
                }

                double c = boss.classifyInstance(i); //classify series i, while ignoring its corresponding histogram i
                if (c == series.get(i).classValue()) {
                    ++correct;
                }
            }
        }

        return (double) correct / (double) numInst;
    }

    private boolean makesItIntoEnsemble(double acc, double maxAcc, double minMaxAcc, int curEnsembleSize) {
        if (acc >= maxAcc * correctThreshold) {
            if (curEnsembleSize >= maxEnsembleSize)
                return acc > minMaxAcc;
            else
                return true;
        }

        return false;
    }

    private double findEnsembleTrainAcc(Instances data) throws Exception {
        this.ensembleCvPreds = new double[data.numInstances()];

        double correct = 0;
        for (int i = 0; i < data.numInstances(); ++i) {
            long predTime = System.nanoTime();
            double[] probs = distributionForInstance(i, data.numClasses());
            predTime = System.nanoTime() - predTime;

            double c = 0;
            for (int j = 1; j < probs.length; j++)
                if (probs[j] > probs[(int) c])
                    c = j;

            //No need to do it againclassifyInstance(i, data.numClasses()); //classify series i, while ignoring its corresponding histogram i
            if (c == data.get(i).classValue())
                ++correct;

            this.ensembleCvPreds[i] = c;

            trainResults.addPrediction(data.get(i).classValue(), probs, c, predTime, "");
        }

        double result = correct / data.numInstances();

        return result;
    }

    public double getTrainAcc(){
        if(ensembleCvAcc>=0){
            return this.ensembleCvAcc;
        }

        try{
            return this.findEnsembleTrainAcc(train);
        }catch(Exception e){
            e.printStackTrace();
        }
        return -1;
    }

    public double[] getTrainPreds(){
        if(this.ensembleCvPreds==null){
            try{
                this.findEnsembleTrainAcc(train);
            }catch(Exception e){
                e.printStackTrace();
            }
        }

        return this.ensembleCvPreds;
    }

    private double[] distributionForInstance(int test, int numclasses) throws Exception {
        double[][] classHist = new double[numSeries][numclasses];

        //get sum of all channels, votes from each are weighted the same.
        double sum[] = new double[numSeries];

        for (int n = 0; n < numSeries; n++) {
            for (BOSSIndividual classifier : classifiers[n]) {
                double classification = classifier.classifyInstance(test);

                classHist[n][(int) classification] += classifier.weight;
                sum[n] += classifier.weight;
            }
        }

        double[] distributions = new double[numclasses];

        for (int n = 0; n < numSeries; n++){
            if (sum[n] != 0)
                for (int i = 0; i < classHist[n].length; ++i)
                    distributions[i] += (classHist[n][i] / sum[n]) / numSeries;
        }

        return distributions;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] dist = distributionForInstance(instance);

        double maxFreq=dist[0], maxClass=0;
        for (int i = 1; i < dist.length; ++i) {
            if (dist[i] > maxFreq) {
                maxFreq = dist[i];
                maxClass = i;
            }
            else if (dist[i] == maxFreq){
                if (rand.nextBoolean()){
                    maxClass = i;
                }
            }
        }

        return maxClass;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[][] classHist = new double[numSeries][instance.numClasses()];

        //get sum of all channels, votes from each are weighted the same.
        double sum[] = new double[numSeries];

        Instance[] series;

        //Multivariate
        if (isMultivariate) {
            series = splitMultivariateInstanceWithClassVal(instance);
        }
        //Univariate
        else {
            series = new Instance[1];
            series[0] = instance;
        }

        if (multiThread){
            ex = Executors.newFixedThreadPool(numThreads);
            ArrayList<BOSSIndividual.TestNearestNeighbourThread> threads = new ArrayList<>(sum(numClassifiers));

            for (int n = 0; n < numSeries; n++) {
                for (BOSSIndividual classifier : classifiers[n]) {
                    BOSSIndividual.TestNearestNeighbourThread t = classifier.new TestNearestNeighbourThread(instance, classifier.weight, n);
                    threads.add(t);
                    ex.execute(t);
                }
            }

            ex.shutdown();
            while (!ex.isTerminated());

            for (BOSSIndividual.TestNearestNeighbourThread t: threads){
                classHist[t.series][(int)t.nn] += t.weight;
                sum[t.series] += t.weight;
            }
        }
        else {
            for (int n = 0; n < numSeries; n++) {
                for (BOSSIndividual classifier : classifiers[n]) {
                    double classification = classifier.classifyInstance(series[n]);
                    classHist[n][(int) classification] += classifier.weight;
                    sum[n] += classifier.weight;
                }
            }
        }

        double[] distributions = new double[instance.numClasses()];

        for (int n = 0; n < numSeries; n++){
            if (sum[n] != 0)
                for (int i = 0; i < classHist[n].length; ++i)
                    distributions[i] += (classHist[n][i] / sum[n]) / numSeries;
        }

        return distributions;
    }

    public static void main(String[] args) throws Exception{
        int fold = 0;

        //Minimum working example
        String dataset = "ItalyPowerDemand";
        Instances train = DatasetLoading.loadDataNullable("Z:\\Data\\TSCProblems2018\\"+dataset+"\\"+dataset+"_TRAIN.arff");
        Instances test = DatasetLoading.loadDataNullable("Z:\\Data\\TSCProblems2018\\"+dataset+"\\"+dataset+"_TEST.arff");
        Instances[] data = resampleTrainAndTestInstances(train, test, fold);
        train = data[0];
        test = data[1];

        String dataset2 = "ERing";
        Instances train2 = DatasetLoading.loadDataNullable("Z:\\Data\\MultivariateTSCProblems\\"+dataset2+"\\"+dataset2+"_TRAIN.arff");
        Instances test2 = DatasetLoading.loadDataNullable("Z:\\Data\\MultivariateTSCProblems\\"+dataset2+"\\"+dataset2+"_TEST.arff");
        Instances[] data2 = resampleMultivariateTrainAndTestInstances(train2, test2, fold);
        train2 = data2[0];
        test2 = data2[1];

        BOSS c;
        double accuracy;

        c = new BOSS();
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("BOSS accuracy on " + dataset + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new BOSS();
        c.buildClassifier(train2);
        accuracy = ClassifierTools.accuracy(test2, c);

        System.out.println("BOSS accuracy on " + dataset2 + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        //Output 22/07/19
        /*
        BOSS accuracy on ItalyPowerDemand fold 0 = 0.9271137026239067 numClassifiers = [4]
        BOSS accuracy on ERing fold 0 = 0.7925925925925926 numClassifiers = [4, 1, 3, 6]
        */
    }
}