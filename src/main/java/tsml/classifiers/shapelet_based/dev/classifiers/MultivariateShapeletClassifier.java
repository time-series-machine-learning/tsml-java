package tsml.classifiers.shapelet_based.dev.classifiers;

import evaluation.evaluators.CrossValidationEvaluator;
import experiments.Experiments;
import machine_learning.classifiers.RidgeClassifierCV;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.TrainTimeContractable;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;

import java.util.concurrent.TimeUnit;

public class MultivariateShapeletClassifier  extends EnhancedAbstractClassifier implements TrainTimeContractable{
    private boolean normalise = true;


    private long trainContractTimeNanos = 0;
    private boolean trainTimeContract = false;

    private Classifier cls = new RidgeClassifierCV();

    private MultivariateShapeletTransformer transformer;
    private Instances header;

    MSTC.ShapeletParams params;
    Experiments.ExperimentalArguments exp;

    public MultivariateShapeletClassifier(Experiments.ExperimentalArguments exp){
       this.exp = exp;
       this.params = new MSTC.ShapeletParams(100,
                3,
                50,
                1000,
                10,
                MSTC.ShapeletFilters.RANDOM, MSTC.ShapeletQualities.GAIN_BINARY,
                MSTC.ShapeletFactories.INDEPENDENT,
                MSTC.AuxClassifiers.ROT);

    }

    @Override
    public String getParameters() {
        return transformer.getParameters();
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
    public void setNormalise(boolean normalise) {
        this.normalise = normalise;
    }
    public void setClassifier(Classifier cls) {
        this.cls = cls;
    }

    @Override
    public void setTrainTimeLimit(long time) {
        trainContractTimeNanos = time;
        trainTimeContract = true;
    }

    @Override
    public boolean withinTrainContract(long start) {
        if (trainContractTimeNanos <= 0) return true; //Not contracted
        return System.nanoTime() - start < trainContractTimeNanos/5*4;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        buildClassifier(Converter.fromArff(data));
    }

    @Override
    public void buildClassifier(TimeSeriesInstances data) throws Exception {


        int n = data.numInstances();
        int m = data.getMinLength()-1;
        int c = data.getMaxNumDimensions();
//        int numShapelets = Math.min(10000,Math.max(500,(int)Math.sqrt(n*m*c)));
        int numShapelets = 10*n < 1000 ? 10*n: 1000;
        this.params.k = numShapelets;
        this.params.max = Math.min(m-1,500);
        this.cls = this.params.classifier.createClassifier();



        trainResults.setBuildTime(System.nanoTime());



        Instances trainEstData = null;


        transformer = new MultivariateShapeletTransformer(params);
        if (seedClassifier) transformer.setSeed(seed);


        Instances transformedData = Converter.toArff(transformer.fitTransform(data));
        header = new Instances(transformedData, 0);

        if (cls instanceof Randomizable) {
           ((Randomizable) cls).setSeed(seed);
        }

        cls.buildClassifier(transformedData);

        if (getEstimateOwnPerformance()) {
            trainEstData = transformedData;
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

    private void estimateOwnPerformance(Instances data) throws Exception {
        int numFolds = Math.min(data.numInstances(), 10);
        CrossValidationEvaluator cv = new CrossValidationEvaluator();
        if (seedClassifier)
            cv.setSeed(seed * 5);
        cv.setNumFolds(numFolds);
        Classifier newCls = AbstractClassifier.makeCopy(cls);
        if (seedClassifier && cls instanceof Randomizable)
            ((Randomizable) newCls).setSeed(seed * 100);
        long tt = trainResults.getBuildTime();
        trainResults = cv.evaluate(newCls, data);
        trainResults.setBuildTime(tt);
        trainResults.setClassifierName("MSTCCV");
        trainResults.setErrorEstimateMethod("CV_" + numFolds);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] probs = distributionForInstance(instance);
        return findIndexOfMax(probs, rand);
    }

    public double[] distributionForInstance(Instance instance) throws Exception {
        Instance transformedInst = transformer.transform(instance);
        transformedInst.setDataset(header);
        return cls.distributionForInstance(transformedInst);
    }
}
