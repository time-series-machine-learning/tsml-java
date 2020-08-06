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
package tsml.classifiers.dictionary_based;

import java.util.*;

import tsml.classifiers.*;

import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import utilities.*;
import weka.core.*;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;

import static utilities.InstanceTools.resampleTrainAndTestInstances;
import static utilities.multivariate_tools.MultivariateInstanceTools.*;

/**
 * BOSS classifier with parameter search and ensembling for univariate and
 * multivariate time series classification.
 * If parameters are known, use the class IndividualBOSS and directly provide them.
 *
 * Alphabetsize fixed to four and maximum wordLength of 16.
 *
 * @author James Large, updated by Matthew Middlehurst
 *
 * Implementation based on the algorithm described in getTechnicalInformation()
 *
 * It is not contractable on tuneable. See cBOSS
 */
public class BOSS extends EnhancedAbstractClassifier implements
        TechnicalInformationHandler, MultiThreadable {

    private transient LinkedList<IndividualBOSS>[] classifiers;
    private int numDimensions;
    private int[] numClassifiers;
    private int currentSeries = 0;
    private boolean isMultivariate = false;

    private final int[] wordLengths = { 16, 14, 12, 10, 8 };
    private final int[] alphabetSize = { 4 };
    private final boolean[] normOptions = { true, false };
    private final double correctThreshold = 0.92;
    private int maxEnsembleSize = 500;

    private transient Instances train;
    private double ensembleCvAcc = -1;
    private double[] ensembleCvPreds = null;

    private int numThreads = 1;
    private boolean multiThread = false;
    private ExecutorService ex;

    protected static final long serialVersionUID = 22554L;
    
    public BOSS() {
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;
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

        sb.append(",numSeries,").append(numDimensions);

        for (int n = 0; n < numDimensions; n++) {
            sb.append(",numclassifiers,").append(n).append(",").append(numClassifiers[n]);

            for (int i = 0; i < numClassifiers[n]; ++i) {
                IndividualBOSS boss = classifiers[n].get(i);
                sb.append(",windowSize,").append(boss.getWindowSize()).append(",wordLength,").append(boss.getWordLength());
                sb.append(",alphabetSize,").append(boss.getAlphabetSize()).append(",norm,").append(boss.isNorm());
            }
        }

        return sb.toString();
    }
    
    @Override
    public void enableMultiThreading(int numThreads) {
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
    public ClassifierResults getTrainResults(){
        return trainResults;
    }

    public void setMaxEnsembleSize(int size) {
        maxEnsembleSize = size;
    }

    @Override
    public void buildClassifier(final Instances data) throws Exception {
        printDebug("Building BOSS");

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
            numDimensions = numDimensions(data);
            classifiers = new LinkedList[numDimensions];

            for (int n = 0; n < numDimensions; n++){
                classifiers[n] = new LinkedList<>();
            }

            numClassifiers = new int[numDimensions];
        }
        //Univariate
        else{
            numDimensions = 1;
            classifiers = new LinkedList[1];
            classifiers[0] = new LinkedList<>();
            numClassifiers = new int[1];
        }

        rand = new Random(seed);

        this.train = data;

        if (multiThread){
            if (numThreads == 1) numThreads = Runtime.getRuntime().availableProcessors();
            if (ex == null) ex = Executors.newFixedThreadPool(numThreads);
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

        for (int n = 0; n < numDimensions; n++) {
            currentSeries = n;
            double maxAcc = -1.0;

            //the acc of the worst member to make it into the final ensemble as it stands
            double minMaxAcc = -1.0;

            for (boolean normalise : normOptions) {
                for (int winSize = minWindow; winSize <= maxWindow; winSize += winInc) {
                    IndividualBOSS boss = new IndividualBOSS(wordLengths[0], alphabetSize[0], winSize, normalise, multiThread, numThreads, ex);
                    boss.seed = seed;
                    boss.buildClassifier(series[n]); //initial setup for this windowsize, with max word length

                    IndividualBOSS bestClassifierForWinSize = null;
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
                    if(classifiers[n].size()%10==0)
                        printLineDebug(" BOSS Model "+(classifiers[n].size()+1)+" found ");
                    //if this window size's accuracy is not good enough to make it into the ensemble, dont bother storing at all
                    if (makesItIntoEnsemble(bestAccForWinSize, maxAcc, minMaxAcc, classifiers[n].size())) {
                        bestClassifierForWinSize.clean();
                        bestClassifierForWinSize.accuracy = bestAccForWinSize;
                        classifiers[n].add(bestClassifierForWinSize);

                        if (bestAccForWinSize > maxAcc) {
                            maxAcc = bestAccForWinSize;
                            //get rid of any extras that dont fall within the new max threshold
                            Iterator<IndividualBOSS> it = classifiers[n].iterator();
                            while (it.hasNext()) {
                                IndividualBOSS b = it.next();
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
        if (getEstimateOwnPerformance()) {
//            trainResults.finaliseResults();
            double result = findEnsembleTrainAcc(data);
//            System.out.println("CV acc ="+result);
        }
        trainResults.setParas(getParameters());
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

    private double individualTrainAcc(IndividualBOSS boss, Instances series, double lowestAcc) throws Exception {
        int correct = 0;
        int numInst = series.numInstances();
        int requiredCorrect = (int)(lowestAcc*numInst);

        if (multiThread){
            ArrayList<Future<Double>> futures = new ArrayList<>(numInst);

            for (int i = 0; i < numInst; ++i)
                futures.add(ex.submit(boss.new TrainNearestNeighbourThread(i)));

            int idx = 0;
            for (Future<Double> f: futures){
                if (f.get() == series.get(idx).classValue()) {
                    ++correct;
                }
                idx++;
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

        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        trainResults.setClassifierName(getClassifierName());
        trainResults.setDatasetName(data.relationName());
        trainResults.setFoldID(seed);
        trainResults.setSplit("train");
        trainResults.setParas(getParameters());
        
        double correct = 0;
        double[] actuals=new double[data.numInstances()];

        for (int i = 0; i < data.numInstances(); ++i) {
            actuals[i]=data.instance(i).classValue();
            long predTime = System.nanoTime();
            // classify series i, while ignoring its corresponding histogram i
            double[] probs = distributionForInstance(i, data.numClasses());
            predTime = System.nanoTime() - predTime;

            int maxClass = findIndexOfMax(probs, rand);
            if (maxClass == data.get(i).classValue())
                ++correct;
            this.ensembleCvPreds[i] = maxClass;

            trainResults.addPrediction(data.get(i).classValue(), probs, maxClass, predTime, "");
        }
        trainResults.finaliseResults(actuals);
        
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

    private double[] distributionForInstance(int test, int numClasses) throws Exception {
        double[] classHist = new double[numClasses];

        //get sum of all channels, votes from each are weighted the same.
        double sum = 0;

        for (int n = 0; n < numDimensions; n++) {
            for (IndividualBOSS classifier : classifiers[n]) {
                double classification = classifier.classifyInstance(test);

                classHist[(int) classification] += classifier.weight;
                sum += classifier.weight;
            }
        }

        double[] distributions = new double[numClasses];

        if (sum != 0) {
            for (int i = 0; i < classHist.length; ++i)
                distributions[i] += (classHist[i] / sum) / numDimensions;
        }
        else{
            for (int i = 0; i < classHist.length; ++i)
                distributions[i] += 1 / numClasses;
        }

        return distributions;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] probs = distributionForInstance(instance);
        return findIndexOfMax(probs, rand);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        int numClasses = train.numClasses();
        double[] classHist = new double[numClasses];

        //get sum of all channels, votes from each are weighted the same.
        double sum = 0;

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
            ArrayList<Future<Double>>[] futures = new ArrayList[numDimensions];

            for (int n = 0; n < numDimensions; n++) {
                futures[n] = new ArrayList<>(numClassifiers[n]);
                for (IndividualBOSS classifier : classifiers[n]) {
                    futures[n].add(ex.submit(classifier.new TestNearestNeighbourThread(series[n])));
                }
            }

            for (int n = 0; n < numDimensions; n++) {
                int idx = 0;
                for (Future<Double> f : futures[n]) {
                    double weight = classifiers[n].get(idx).weight;
                    classHist[f.get().intValue()] += weight;
                    sum += weight;
                    idx++;
                }
            }
        }
        else {
            for (int n = 0; n < numDimensions; n++) {
                for (IndividualBOSS classifier : classifiers[n]) {
                    double classification = classifier.classifyInstance(series[n]);
                    classHist[(int) classification] += classifier.weight;
                    sum += classifier.weight;
                }
            }
        }

        double[] distributions = new double[instance.numClasses()];

        if (sum != 0) {
            for (int i = 0; i < classHist.length; ++i)
                distributions[i] += (classHist[i] / sum);
        }
        else{
            for (int i = 0; i < classHist.length; ++i)
                distributions[i] += 1 / numClasses;
        }

        return distributions;
    }

    public static void main(String[] args) throws Exception{
        int fold = 0;

        //Minimum working example
        String dataset = "ItalyPowerDemand";
        Instances train = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\"+dataset+"\\"+dataset+"_TRAIN.arff");
        Instances test = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\"+dataset+"\\"+dataset+"_TEST.arff");
        Instances[] data = resampleTrainAndTestInstances(train, test, fold);
        train = data[0];
        test = data[1];

        String dataset2 = "ERing";
        Instances train2 = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Multivariate_arff\\"+dataset2+"\\"+dataset2+"_TRAIN.arff");
        Instances test2 = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Multivariate_arff\\"+dataset2+"\\"+dataset2+"_TEST.arff");
        Instances[] data2 = resampleMultivariateTrainAndTestInstances(train2, test2, fold);
        train2 = data2[0];
        test2 = data2[1];

        BOSS c;
        double accuracy;

        c = new BOSS();
        c.setEstimateOwnPerformance(true);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("BOSS accuracy on " + dataset + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new BOSS();
        c.setEstimateOwnPerformance(true);
        c.buildClassifier(train2);
        accuracy = ClassifierTools.accuracy(test2, c);

        System.out.println("BOSS accuracy on " + dataset2 + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        //Output 02/08/19
        /*
        CV acc =0.9402985074626866
        BOSS accuracy on ItalyPowerDemand fold 0 = 0.9271137026239067 numClassifiers = [4]
        CV acc =0.8333333333333334
        BOSS accuracy on ERing fold 0 = 0.8333333333333334 numClassifiers = [4, 1, 3, 6]
        */
    }
}