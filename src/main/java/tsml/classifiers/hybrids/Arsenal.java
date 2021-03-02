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
public class Arsenal extends EnhancedAbstractClassifier implements TrainTimeContractable, MultiThreadable {

    private int numKernels = 2000;
    private int ensembleSize = 25;
    private boolean normalise = true;
    private Classifier cls = new RidgeClassifierCV();

    private long trainContractTimeNanos = 0;
    private boolean trainTimeContract = false;

    private boolean multithreading = false;
    private int threads;

    private Classifier[] classifiers;
    private ROCKET[] rockets;
    private Instances header;

    public Arsenal(){
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
    }

    @Override
    public String getParameters() {
        int cl = classifiers == null ? 0 : classifiers.length;
        String temp=super.getParameters()+",numKernels,"+numKernels+",normalise,"+normalise+",ensembleSize,"+
                cl+ ",trainContract,"+trainTimeContract+",contractTime,"+trainContractTimeNanos;
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

    public void setEnsembleSize(int ensembleSize) { this.ensembleSize = ensembleSize; }

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

        Instances[] trainEstData = null;

        if (trainTimeContract) {
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
                if (getEstimateOwnPerformance()) {
                    transformedDataArr.add(transformedData);
                }
                i++;
            }

            classifiers = new Classifier[tempCls.size()];
            classifiers = tempCls.toArray(classifiers);
            rockets = new ROCKET[tempROCKET.size()];
            rockets = tempROCKET.toArray(rockets);

            if (getEstimateOwnPerformance()) {
                trainEstData = new Instances[transformedDataArr.size()];
                trainEstData = transformedDataArr.toArray(trainEstData);
            }
        }
        else {
            classifiers = new Classifier[ensembleSize];
            rockets = new ROCKET[ensembleSize];
            trainEstData = new Instances[ensembleSize];
            for (int i = 0; i < ensembleSize; i++) {
                rockets[i] = new ROCKET();
                rockets[i].setNumKernels(numKernels);
                rockets[i].setNormalise(normalise);
                rockets[i].setSeed(seed+i*47);

                if (multithreading) {
                    rockets[i].enableMultiThreading(threads);
                }

                Instances transformedData = rockets[i].fitTransform(data);
                if (header == null) header = new Instances(transformedData, 0);

                classifiers[i] = AbstractClassifier.makeCopy(cls);

                if (classifiers[i] instanceof Randomizable) {
                    ((Randomizable) classifiers[i]).setSeed(seed+i*47);
                }

                classifiers[i].buildClassifier(transformedData);

                if (getEstimateOwnPerformance()) {
                    trainEstData[i] = transformedData;
                }
            }
        }

        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        trainResults.setBuildTime(System.nanoTime() - trainResults.getBuildTime());
        if (getEstimateOwnPerformance()) {
            long est1 = System.nanoTime();
            estimateOwnPerformance(trainEstData);
            long est2 = System.nanoTime();
            trainResults.setErrorEstimateTime(est2 - est1 + trainResults.getErrorEstimateTime());
        }
        trainResults.setBuildPlusEstimateTime(trainResults.getBuildTime() + trainResults.getErrorEstimateTime());
        trainResults.setParas(getParameters());
    }

    private void estimateOwnPerformance(Instances[] data) throws Exception {
        if (estimator == EstimatorMethod.CV || estimator == EstimatorMethod.NONE) {
            double[] preds=new double[data[0].numInstances()];
            double[] actuals=new double[data[0].numInstances()];
            double[][] trainDistributions=new double[data[0].numInstances()][data[0].numClasses()];
            long[] predTimes=new long[data[0].numInstances()];//Dummy variable, need something
            int numFolds = Math.min(data[0].numInstances(), 10);
            for (int r = 0; r < data.length; r++) {
                CrossValidationEvaluator cv = new CrossValidationEvaluator();
                if (seedClassifier)
                    cv.setSeed(seed*5*(r+1));
                cv.setNumFolds(numFolds);
                Classifier newCls = AbstractClassifier.makeCopy(cls);
                if (seedClassifier && cls instanceof Randomizable)
                    ((Randomizable)newCls).setSeed(seed*100*(r+1));
                ClassifierResults results = cv.evaluate(newCls, data[r]);
                for (int i = 0; i < preds.length; i++) {
                    double[] dist = results.getProbabilityDistribution(i);
                    for (int n = 0; n < trainDistributions[i].length; n++){
                        trainDistributions[i][n] += dist[n];
                    }
                }
            }
            for (int i = 0; i < preds.length; i++){
                preds[i] = findIndexOfMax(trainDistributions[i], rand);
                actuals[i] = data[0].get(i).classValue();
                for (int n = 0; n < trainDistributions[i].length; n++) {
                    trainDistributions[i][n] /= data.length;
                }
            }
            trainResults.addAllPredictions(actuals,preds,trainDistributions,predTimes, null);
            trainResults.setDatasetName(data[0].relationName());
            trainResults.setSplit("train");
            trainResults.setFoldID(seed);
            trainResults.setClassifierName("ArsenalCV");
            trainResults.setErrorEstimateMethod("CV_" + numFolds);
        } else if (estimator == EstimatorMethod.OOB) {
            int[] oobCount = new int[data[0].numInstances()];
            double[] preds=new double[data[0].numInstances()];
            double[] actuals=new double[data[0].numInstances()];
            double[][] trainDistributions=new double[data[0].numInstances()][data[0].numClasses()];
            long[] predTimes=new long[data[0].numInstances()];//Dummy variable, need something
            for (int r = 0; r < data.length; r++) {
                OutOfBagEvaluator oob = new OutOfBagEvaluator();
                oob.setSeed((seed+1)*5*(r+1));
                Classifier newCls = AbstractClassifier.makeCopy(cls);
                if (seedClassifier && cls instanceof Randomizable)
                    ((Randomizable)newCls).setSeed((seed+1)*100*(r+1));
                ClassifierResults results = oob.evaluate(newCls, data[r]);
                List<Integer> indicies = oob.getOutOfBagTestDataIndices();
                for (int i = 0; i < indicies.size(); i++) {
                    int index = indicies.get(i);
                    oobCount[index]++;
                    double[] dist = results.getProbabilityDistribution(i);
                    for (int n = 0; n < trainDistributions[i].length; n++) {
                        trainDistributions[index][n] += dist[n];
                    }
                }
            }
            for (int i = 0; i < preds.length; i++) {
                if (oobCount[i] > 0) {
                    preds[i] = findIndexOfMax(trainDistributions[i], rand);
                    for (int n = 0; n < trainDistributions[i].length; n++) {
                        trainDistributions[i][n] /= oobCount[i];
                    }
                }
                else{
                    Arrays.fill(trainDistributions[i], 1.0/numClasses);
                }

                actuals[i] = data[0].get(i).classValue();
                preds[i] = findIndexOfMax(trainDistributions[i], rand);
            }
            trainResults.addAllPredictions(actuals,preds,trainDistributions,predTimes, null);
            trainResults.setDatasetName(data[0].relationName());
            trainResults.setSplit("train");
            trainResults.setFoldID(seed);
            trainResults.setClassifierName("ArsenalOOB");
            trainResults.setErrorEstimateMethod("OOB");
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] probs = distributionForInstance(instance);
        return findIndexOfMax(probs, rand);
    }

    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] probs = new double[header.numClasses()];
        double sum = 0;
        for (int i = 0; i < classifiers.length; i++){
            Instance transformedInst = rockets[i].transform(instance);
            transformedInst.setDataset(header);
            double pls = classifiers[i].classifyInstance(transformedInst);
            double s = cls instanceof RidgeClassifierCV ?
                    Math.pow(((RidgeClassifierCV) classifiers[i]).getBestScore(), 4) : 1;
            probs[(int)pls] += s;
            sum += s;
        }

        for (int i = 0; i < probs.length; i++) probs[i] /= sum;
        return probs;
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

        Arsenal c;
        double accuracy;

        c = new Arsenal();
        c.seed = fold;
        c.setEstimateOwnPerformance(true);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("Arsenal accuracy on " + dataset + " fold " + fold + " = " + accuracy);
        System.out.println("Train accuracy on " + dataset + " fold " + fold + " = " + c.trainResults.getAcc());
        System.out.println("Build time on " + dataset + " fold " + fold + " = " +
                TimeUnit.SECONDS.convert(c.trainResults.getBuildTime(), TimeUnit.NANOSECONDS) + " seconds");

        c = new Arsenal();
        c.seed = fold;
        c.buildClassifier(train2);
        accuracy = ClassifierTools.accuracy(test2, c);

        System.out.println("Arsenal accuracy on " + dataset2 + " fold " + fold + " = " + accuracy);
        System.out.println("Build time on " + dataset2 + " fold " + fold + " = " +
                TimeUnit.SECONDS.convert(c.trainResults.getBuildTime(), TimeUnit.NANOSECONDS) + " seconds");
    }
}
