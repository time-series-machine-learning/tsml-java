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
package tsml.classifiers.hybrids;

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

    public int numKernels = 10000;
    private boolean normalise = true;
    private Classifier cls = new RidgeClassifierCV();

    private ROCKET rocket;
    private Instances header;

    private long trainContractTimeNanos;
    private boolean trainTimeContract = false;
    private int numKernelsStep = 50;

    private boolean multithreading = false;
    private int threads;

    boolean ensemble = true;
    public int ensembleSize = 25;
    Classifier[] eCls;
    ROCKET[] eROCKET;

    public ROCKETClassifier(){
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
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

        if (trainTimeContract) {
            if (ensemble) {
                ArrayList<Classifier> tempCls = new ArrayList<>();
                ArrayList<ROCKET> tempROCKET = new ArrayList<>();
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
                    i++;
                }

                eCls = tempCls.toArray(eCls);
                eROCKET = tempROCKET.toArray(eROCKET);
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
            }
        }
        else {
            if (ensemble){
                eCls = new Classifier[ensembleSize];
                eROCKET = new ROCKET[ensembleSize];
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
            }
        }

        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        trainResults.setBuildTime(System.nanoTime() - trainResults.getBuildTime());
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
                double s = cls instanceof RidgeClassifierCV ? Math.pow(((RidgeClassifierCV)eCls[i]).bestScore, 4) : 1;
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
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("ROCKETClassifier accuracy on " + dataset + " fold " + fold + " = " + accuracy);
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
    }
}
