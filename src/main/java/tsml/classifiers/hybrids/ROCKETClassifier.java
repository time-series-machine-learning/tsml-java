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

    private long trainContractTimeNanos = 0;
    private boolean trainTimeContract = false;
    private int numKernelsStep = 50;

    private boolean multithreading = false;
    private int threads;

    private ROCKET rocket;
    private Instances header;

    public ROCKETClassifier(){
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
    }

    @Override
    public String getParameters() {
        int nc = rocket == null ? numKernels : rocket.getNumKernels();
        String temp=super.getParameters()+",numKernels,"+nc+",normalise,"+normalise+",trainContract,"+
                trainTimeContract+",contractTime,"+trainContractTimeNanos+ ",numKernelStep,"+ numKernelsStep;
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

        if (trainTimeContract) {
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

        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        trainResults.setBuildTime(System.nanoTime() - trainResults.getBuildTime());
        if(getEstimateOwnPerformance()){
            long est1 = System.nanoTime();
            estimateOwnPerformance(trainEstData);
            long est2 = System.nanoTime();
            trainResults.setErrorEstimateTime(est2 - est1 + trainResults.getErrorEstimateTime());
        }
        trainResults.setBuildPlusEstimateTime(trainResults.getBuildTime() + trainResults.getErrorEstimateTime());
        trainResults.setParas(getParameters());
    }

    private void estimateOwnPerformance(Instances data) throws Exception {
        int numFolds=Math.min(data.numInstances(), 10);
        CrossValidationEvaluator cv = new CrossValidationEvaluator();
        if (seedClassifier)
            cv.setSeed(seed*5);
        cv.setNumFolds(numFolds);
        Classifier newCls = AbstractClassifier.makeCopy(cls);
        if (seedClassifier && cls instanceof Randomizable)
            ((Randomizable)newCls).setSeed(seed*100);
        long tt = trainResults.getBuildTime();
        trainResults=cv.evaluate(newCls, data);
        trainResults.setBuildTime(tt);
        trainResults.setClassifierName("ROCKETCV");
        trainResults.setErrorEstimateMethod("CV_"+numFolds);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] probs = distributionForInstance(instance);
        return findIndexOfMax(probs, rand);
    }

    public double[] distributionForInstance(Instance instance) throws Exception {
        Instance transformedInst = rocket.transform(instance);
        transformedInst.setDataset(header);
        return cls.distributionForInstance(transformedInst);
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
    }
}
