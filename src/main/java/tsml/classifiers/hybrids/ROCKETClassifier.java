/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.classifiers.hybrids;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.OutOfBagEvaluator;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import machine_learning.classifiers.RidgeClassifierCV;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.MultiThreadable;
import tsml.classifiers.TrainTimeContractable;
import tsml.transformers.ROCKET;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static utilities.InstanceTools.resampleTrainAndTestInstances;
import static utilities.multivariate_tools.MultivariateInstanceTools.resampleMultivariateTrainAndTestInstances;

/**
 * Contractable classifier making use of the ROCKET transformer.
 *
 * Transform based on sktime python implementation by the author:
 * https://github.com/alan-turing-institute/sktime/blob/master/sktime/transformers/series_as_features/rocket.py
 *
 * @author Matthew Middlehurst
 */
public class ROCKETClassifier extends EnhancedAbstractClassifier implements TrainTimeContractable, MultiThreadable {

    private int numKernels = 10000;
    private boolean normalise = true;
    private Classifier cls = new RidgeClassifierCV();

    private ROCKET rocket;
    private Instances header;

    private long trainContractTimeNanos = 0;
    private boolean trainTimeContract = false;
    private int numKernelsStep = 50;

    private boolean multithreading = false;
    private int threads;

    private boolean ensemble = false;
    private int ensembleSize = 25;
    private Classifier[] eCls;
    private ROCKET[] eROCKET;

    public ROCKETClassifier(){
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
    }

    @Override
    public String getParameters() {
        int nc = numKernels;
        if (rocket != null) nc = rocket.getNumKernels();
        int es = 0;
        if (eCls != null) es = eCls.length;
        String temp=super.getParameters()+",numKernels,"+nc+",normalise,"+normalise+",ensemble,"+ensemble+
                ",ensembleSize,"+es+ ",trainContract,"+trainTimeContract+",contractTime,"+trainContractTimeNanos+
                ",numKernelStep,"+ numKernelsStep;
        return temp;
    }

    @Override
    public Capabilities getCapabilities() {
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

    public void setNumKernels(int numKernels){ this.numKernels = numKernels; }

    public void setNormalise(boolean normalise){
        this.normalise = normalise;
    }

    public void setClassifier(Classifier cls){
        this.cls = cls;
    }

    public void setEnsemble(boolean ensemble) { this.ensemble = ensemble; }

    public void setEnsembleSize(int ensembleSize) { this.ensembleSize = ensembleSize; }

    public void setNumKernelsStep(int numKernelsStep) { this.numKernelsStep = numKernelsStep; }

    @Override
    public void setTrainTimeLimit(long time) {
        trainContractTimeNanos = time;
        trainTimeContract = true;
    }

    @Override
    public boolean withinTrainContract(long start) {
        if(trainContractTimeNanos <= 0) return true; //Not contracted
        return System.nanoTime() - start < trainContractTimeNanos;
    }

    @Override
    public void enableMultiThreading(int numThreads){
        multithreading = true;
        threads = numThreads;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        super.buildClassifier(data);
        trainResults.setBuildTime(System.nanoTime());
        getCapabilities().testWithFail(data);

        if (multithreading && cls instanceof MultiThreadable)
            ((MultiThreadable)cls).enableMultiThreading(threads);

        Instances trainEstData = null;
        Instances[] ensembleTrainEstData = null;

        if (trainTimeContract) {
            if (ensemble) {
                ArrayList<Classifier> tempCls = new ArrayList<>();
                ArrayList<ROCKET> tempROCKET = new ArrayList<>();
                ArrayList<Instances> transformedDataArr = new ArrayList<>();
                int i = 0;
                while (withinTrainContract(trainResults.getBuildTime())) {
                    ROCKET r = new ROCKET();
                    r.setNumKernels(numKernels);
                    r.setNormalise(normalise);
                    r.setSeed(seed+i*47);

                    if (multithreading) {
                        r.enableMultiThreading(threads);
                    }

                    Instances transformedData = r.fitTransform(data);
                    if (header == null) header = new Instances(transformedData, 0);

                    Classifier c = AbstractClassifier.makeCopy(cls);

                    if (c instanceof Randomizable) {
                        ((Randomizable) c).setSeed(seed+i*47);
                    }

                    c.buildClassifier(transformedData);

                    tempCls.add(c);
                    tempROCKET.add(r);
                    if (getEstimateOwnPerformance()){
                        transformedDataArr.add(transformedData);
                    }
                    i++;
                }

                eCls = new Classifier[tempCls.size()];
                eCls = tempCls.toArray(eCls);
                eROCKET = new ROCKET[tempROCKET.size()];
                eROCKET = tempROCKET.toArray(eROCKET);

                if (getEstimateOwnPerformance()){
                    ensembleTrainEstData = new Instances[transformedDataArr.size()];
                    ensembleTrainEstData = transformedDataArr.toArray(ensembleTrainEstData);
                }
            }
            else {
                ArrayList<Instances> fragmentedTransformedData = new ArrayList<>();
                rocket = new ROCKET();
                rocket.setNumKernels(0);
                rocket.setNormalise(normalise);
                rocket.setSeed(seed);

                int l = 0;
                while (withinTrainContract(trainResults.getBuildTime())) {
                    ROCKET tempRocket = new ROCKET();
                    tempRocket.setNumKernels(numKernelsStep);
                    tempRocket.setNormalise(normalise);
                    tempRocket.setSeed(seed + l * numKernelsStep);

                    if (multithreading) {
                        tempRocket.enableMultiThreading(threads);
                    }

                    fragmentedTransformedData.add(tempRocket.fitTransform(data));
                    rocket.addKernels(tempRocket);

                    l++;
                }

                Instances transformedData = rocket.determineOutputFormat(data);
                header = new Instances(transformedData, 0);

                for (int i = 0; i < data.numInstances(); i++) {
                    double[] arr = new double[transformedData.numAttributes()];
                    int a1 = 0;
                    for (Instances insts : fragmentedTransformedData) {
                        Instance inst = insts.get(i);
                        for (int j = 0; j < numKernelsStep * 2; j++) {
                            arr[a1 + j] = inst.value(j);
                        }
                        a1 += numKernelsStep * 2;
                    }
                    arr[arr.length - 1] = data.get(i).classValue();
                    transformedData.add(new DenseInstance(1, arr));
                }

                cls.buildClassifier(transformedData);

                if (getEstimateOwnPerformance()){
                    trainEstData = transformedData;
                }
            }
        }
        else {
            if (ensemble){
                eCls = new Classifier[ensembleSize];
                eROCKET = new ROCKET[ensembleSize];
                ensembleTrainEstData = new Instances[ensembleSize];
                for (int i = 0; i < ensembleSize; i++){
                    eROCKET[i] = new ROCKET();
                    eROCKET[i].setNumKernels(numKernels);
                    eROCKET[i].setNormalise(normalise);
                    eROCKET[i].setSeed(seed+i*47);

                    if (multithreading) {
                        eROCKET[i].enableMultiThreading(threads);
                    }

                    Instances transformedData = eROCKET[i].fitTransform(data);
                    if (header == null) header = new Instances(transformedData, 0);

                    eCls[i] = AbstractClassifier.makeCopy(cls);

                    if (eCls[i] instanceof Randomizable) {
                        ((Randomizable) eCls[i]).setSeed(seed+i*47);
                    }

                    eCls[i].buildClassifier(transformedData);

                    if (getEstimateOwnPerformance()){
                        ensembleTrainEstData[i] = transformedData;
                    }
                }
            }
            else {
                rocket = new ROCKET();
                rocket.setNumKernels(numKernels);
                rocket.setNormalise(normalise);
                rocket.setSeed(seed);

                if (multithreading) {
                    rocket.enableMultiThreading(threads);
                }

                Instances transformedData = rocket.fitTransform(data);
                header = new Instances(transformedData, 0);

                if (cls instanceof Randomizable) {
                    ((Randomizable) cls).setSeed(seed);
                }

                cls.buildClassifier(transformedData);

                if (getEstimateOwnPerformance()){
                    trainEstData = transformedData;
                }
            }
        }

        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        trainResults.setBuildTime(System.nanoTime() - trainResults.getBuildTime());

        if(getEstimateOwnPerformance()){
            long est1 = System.nanoTime();
            estimateOwnPerformance(trainEstData, ensembleTrainEstData);
            long est2 = System.nanoTime();
            trainResults.setErrorEstimateTime(est2 - est1 + trainResults.getErrorEstimateTime());
        }
        trainResults.setBuildPlusEstimateTime(trainResults.getBuildTime() + trainResults.getErrorEstimateTime());
        trainResults.setParas(getParameters());
    }

    private void estimateOwnPerformance(Instances singleData, Instances[] ensembleData) throws Exception {
        if (ensemble) {
            if (estimator == EstimatorMethod.CV || estimator == EstimatorMethod.NONE) {
                double[] preds=new double[ensembleData[0].numInstances()];
                double[] actuals=new double[ensembleData[0].numInstances()];
                double[][] trainDistributions=new double[ensembleData[0].numInstances()][ensembleData[0].numClasses()];
                long[] predTimes=new long[ensembleData[0].numInstances()];//Dummy variable, need something
                int numFolds = Math.min(ensembleData[0].numInstances(), 10);
                for (int r = 0; r < ensembleData.length; r++) {
                    CrossValidationEvaluator cv = new CrossValidationEvaluator();
                    if (seedClassifier)
                        cv.setSeed(seed*5*(r+1));
                    cv.setNumFolds(numFolds);
                    Classifier newCls = AbstractClassifier.makeCopy(cls);
                    if (seedClassifier && cls instanceof Randomizable)
                        ((Randomizable)newCls).setSeed(seed*100*(r+1));
                    ClassifierResults results = cv.evaluate(newCls, ensembleData[r]);
                    for (int i = 0; i < preds.length; i++){
                        double[] dist = results.getProbabilityDistribution(i);
                        for (int n = 0; n < trainDistributions[i].length; n++){
                            trainDistributions[i][n] += dist[n];
                        }
                    }
                }
                for (int i = 0; i < preds.length; i++){
                    preds[i] = findIndexOfMax(trainDistributions[i], rand);
                    actuals[i] = ensembleData[0].get(i).classValue();
                    for (int n = 0; n < trainDistributions[i].length; n++){
                        trainDistributions[i][n] /= ensembleData.length;
                    }
                }
                trainResults.addAllPredictions(actuals,preds,trainDistributions,predTimes, null);
                trainResults.setDatasetName(ensembleData[0].relationName());
                trainResults.setSplit("train");
                trainResults.setFoldID(seed);
                trainResults.setClassifierName("ROCKET-ECV");
                trainResults.setErrorEstimateMethod("CV_" + numFolds);
            } else if (estimator == EstimatorMethod.OOB) {
                int[] oobCount = new int[ensembleData[0].numInstances()];
                double[] preds=new double[ensembleData[0].numInstances()];
                double[] actuals=new double[ensembleData[0].numInstances()];
                double[][] trainDistributions=new double[ensembleData[0].numInstances()][ensembleData[0].numClasses()];
                long[] predTimes=new long[ensembleData[0].numInstances()];//Dummy variable, need something
                for (int r = 0; r < ensembleData.length; r++) {
                    OutOfBagEvaluator oob = new OutOfBagEvaluator();
                    oob.setSeed((seed+1)*5*(r+1));
                    Classifier newCls = AbstractClassifier.makeCopy(cls);
                    if (seedClassifier && cls instanceof Randomizable)
                        ((Randomizable)newCls).setSeed((seed+1)*100*(r+1));
                    ClassifierResults results = oob.evaluate(newCls, ensembleData[r]);
                    List<Integer> indicies = oob.getOutOfBagTestDataIndices();
                    for (int i = 0; i < indicies.size(); i++){
                        int index = indicies.get(i);
                        oobCount[index]++;
                        double[] dist = results.getProbabilityDistribution(i);
                        for (int n = 0; n < trainDistributions[i].length; n++){
                            trainDistributions[index][n] += dist[n];
                        }
                    }
                }
                for (int i = 0; i < preds.length; i++){
                    if (oobCount[i] > 0) {
                        preds[i] = findIndexOfMax(trainDistributions[i], rand);
                        for (int n = 0; n < trainDistributions[i].length; n++) {
                            trainDistributions[i][n] /= oobCount[i];
                        }
                    }
                    else{
                        Arrays.fill(trainDistributions[i], 1.0/numClasses);
                    }

                    actuals[i] = ensembleData[0].get(i).classValue();
                    preds[i] = findIndexOfMax(trainDistributions[i], rand);
                }
                trainResults.addAllPredictions(actuals,preds,trainDistributions,predTimes, null);
                trainResults.setDatasetName(ensembleData[0].relationName());
                trainResults.setSplit("train");
                trainResults.setFoldID(seed);
                trainResults.setClassifierName("ROCKET-EOOB");
                trainResults.setErrorEstimateMethod("OOB");
            }
        }
        else{
            int numFolds=Math.min(singleData.numInstances(), 10);
            CrossValidationEvaluator cv = new CrossValidationEvaluator();
            if (seedClassifier)
                cv.setSeed(seed*5);
            cv.setNumFolds(numFolds);
            Classifier newCls = AbstractClassifier.makeCopy(cls);
            if (seedClassifier && cls instanceof Randomizable)
                ((Randomizable)newCls).setSeed(seed*100);
            long tt = trainResults.getBuildTime();
            trainResults=cv.evaluate(newCls, singleData);
            trainResults.setBuildTime(tt);
            trainResults.setClassifierName("ROCKETCV");
            trainResults.setErrorEstimateMethod("CV_"+numFolds);
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] probs = distributionForInstance(instance);
        return findIndexOfMax(probs, rand);
    }

    public double[] distributionForInstance(Instance instance) throws Exception {
        if (ensemble){
            double[] probs = new double[header.numClasses()];
            double sum = 0;
            for (int i = 0; i < eCls.length; i++){
                Instance transformedInst = eROCKET[i].transform(instance);
                transformedInst.setDataset(header);
                double pls = eCls[i].classifyInstance(transformedInst);
                double s = cls instanceof RidgeClassifierCV ? Math.pow(((RidgeClassifierCV)eCls[i]).getBestScore(), 4)
                        : 1;
                probs[(int)pls] += s;
                sum += s;
            }

            for (int i = 0; i < probs.length; i++) probs[i] /= sum;
            return probs;
        }
        else {
            Instance transformedInst = rocket.transform(instance);
            transformedInst.setDataset(header);
            return cls.distributionForInstance(transformedInst);
        }
    }

    public static void main(String[] args) throws Exception {
        int fold = 0;

        String dataset = "ItalyPowerDemand";
        Instances train = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset
                + "\\" + dataset + "_TRAIN.arff");
        Instances test = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset
                + "\\" + dataset + "_TEST.arff");
        Instances[] data = resampleTrainAndTestInstances(train, test, fold);
        train = data[0];
        test = data[1];

        String dataset2 = "ERing";
        Instances train2 = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Multivariate_arff\\"+dataset2+
                "\\"+dataset2+"_TRAIN.arff");
        Instances test2 = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Multivariate_arff\\"+dataset2+
                "\\"+dataset2+"_TEST.arff");
        Instances[] data2 = resampleMultivariateTrainAndTestInstances(train2, test2, fold);
        train2 = data2[0];
        test2 = data2[1];

        ROCKETClassifier c;
        double accuracy;

        c = new ROCKETClassifier();
        c.seed = fold;
        c.setEstimateOwnPerformance(true);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("ROCKETClassifier accuracy on " + dataset + " fold " + fold + " = " + accuracy);
        System.out.println("Train accuracy on " + dataset + " fold " + fold + " = " + c.trainResults.getAcc());
        System.out.println("Build time on " + dataset + " fold " + fold + " = " +
                TimeUnit.SECONDS.convert(c.trainResults.getBuildTime(), TimeUnit.NANOSECONDS) + " seconds");

        c = new ROCKETClassifier();
        c.seed = fold;
        c.buildClassifier(train2);
        accuracy = ClassifierTools.accuracy(test2, c);

        System.out.println("ROCKETClassifier accuracy on " + dataset2 + " fold " + fold + " = " + accuracy);
        System.out.println("Build time on " + dataset2 + " fold " + fold + " = " +
                TimeUnit.SECONDS.convert(c.trainResults.getBuildTime(), TimeUnit.NANOSECONDS) + " seconds");

        c = new ROCKETClassifier();
        c.seed = fold;
        c.enableMultiThreading(4);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("ROCKETClassifierMT accuracy on " + dataset + " fold " + fold + " = " + accuracy);
        System.out.println("Build time on " + dataset + " fold " + fold + " = " +
                TimeUnit.SECONDS.convert(c.trainResults.getBuildTime(), TimeUnit.NANOSECONDS) + " seconds");

        c = new ROCKETClassifier();
        c.seed = fold;
        c.setTrainTimeLimit(400, TimeUnit.MILLISECONDS);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("ROCKETClassifierContract accuracy on " + dataset + " fold " + fold + " = " + accuracy);
        System.out.println("No Kernels = " + c.rocket.getNumKernels());
        System.out.println("Build time on " + dataset + " fold " + fold + " = " +
                TimeUnit.SECONDS.convert(c.trainResults.getBuildTime(), TimeUnit.NANOSECONDS) + " seconds");

        c = new ROCKETClassifier();
        c.seed = fold;
        c.ensemble = true;
        c.setNumKernels(2000);
        c.setEstimateOwnPerformance(true);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("ROCKETEnsembleClassifier accuracy on " + dataset + " fold " + fold + " = " + accuracy);
        System.out.println("Train accuracy on " + dataset + " fold " + fold + " = " + c.trainResults.getAcc());
        System.out.println("Build time on " + dataset + " fold " + fold + " = " +
                TimeUnit.SECONDS.convert(c.trainResults.getBuildTime(), TimeUnit.NANOSECONDS) + " seconds");

        c = new ROCKETClassifier();
        c.seed = fold;
        c.ensemble = true;
        c.setNumKernels(2000);
        c.buildClassifier(train2);
        accuracy = ClassifierTools.accuracy(test2, c);

        System.out.println("ROCKETEnsembleClassifier accuracy on " + dataset2 + " fold " + fold + " = " + accuracy);
        System.out.println("Build time on " + dataset2 + " fold " + fold + " = " +
                TimeUnit.SECONDS.convert(c.trainResults.getBuildTime(), TimeUnit.NANOSECONDS) + " seconds");

        c = new ROCKETClassifier();
        c.seed = fold;
        c.ensemble = true;
        c.setNumKernels(2000);
        c.enableMultiThreading(4);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("ROCKETEnsembleClassifierMT accuracy on " + dataset + " fold " + fold + " = " + accuracy);
        System.out.println("Build time on " + dataset + " fold " + fold + " = " +
                TimeUnit.SECONDS.convert(c.trainResults.getBuildTime(), TimeUnit.NANOSECONDS) + " seconds");

        c = new ROCKETClassifier();
        c.seed = fold;
        c.ensemble = true;
        c.setNumKernels(2000);
        c.setTrainTimeLimit(400, TimeUnit.MILLISECONDS);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("ROCKETEnsembleClassifierContract accuracy on " + dataset + " fold " + fold + " = " + accuracy);
        System.out.println("No Classifiers = " + c.eCls.length);
        System.out.println("Build time on " + dataset + " fold " + fold + " = " +
                TimeUnit.SECONDS.convert(c.trainResults.getBuildTime(), TimeUnit.NANOSECONDS) + " seconds");
    }
}
