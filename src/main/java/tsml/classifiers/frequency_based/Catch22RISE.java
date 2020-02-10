package tsml.classifiers.frequency_based;
import evaluation.evaluators.SingleSampleEvaluator;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLists;
import scala.util.control.Exception;
import tsml.transformers.Catch22;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.RandomTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

import static experiments.data.DatasetLoading.loadDataNullable;

public class Catch22RISE extends cRISE {
    Catch22 catch22 = null;

    public Catch22RISE(){
        super();
        catch22 = new Catch22();
    }

    @Override
    public void buildClassifier(Instances trainingData) throws java.lang.Exception {

        if(serialisePath != null){
            cRISE temp = readSerialise(seed);
            if(temp != null) {
                this.copyFromSerObject(temp);
                this.loadedFromFile = true;
            }
        }

        //If not loaded from file e.g. Starting fresh experiment.
        if (!loadedFromFile) {
            //Just used for getParameters.
            data = trainingData;
            //(re)Initialise all variables to account for multiple calls of buildClassifier.
            initialise();

            //Check min & max interval lengths are valid.
            if(maxIntervalLength > trainingData.numAttributes()-1 || maxIntervalLength <= 0){
                maxIntervalLength = trainingData.numAttributes()-1;
            }
            if(minIntervalLength >= trainingData.numAttributes()-1 || minIntervalLength <= 0){
                minIntervalLength = (trainingData.numAttributes()-1)/2;
            }

        }

        //Start forest timer.
        timer.forestStartTime = System.nanoTime();

        for (; treeCount < numTrees && (System.nanoTime() - timer.forestStartTime) < (timer.forestTimeLimit - getTime()); treeCount++) {

            //Start tree timer.
            timer.treeStartTime = System.nanoTime();

            //Compute maximum interval length given time remaining.
            timer.buildModel();
            maxIntervalLength = (int)timer.getFeatureSpace((timer.forestTimeLimit) - (System.nanoTime() - (timer.forestStartTime - getTime())));


            //Produce intervalInstances from trainingData using interval attributes.
            Instances intervalInstances;
            //intervalInstances = produceIntervalInstances(maxIntervalLength, trainingData);
            intervalInstances = produceIntervalInstancesUpdate(maxIntervalLength, trainingData);

            //Transform instances.
            if (transformType != null) {
                intervalInstances = transformInstances(intervalInstances, transformType);
                intervalInstances = catch22TransformInstances(intervalInstances);
            }

            //Add independent variable to model (length of interval).
            timer.makePrediciton(intervalInstances.numAttributes() - 1);
            timer.independantVariables.add(intervalInstances.numAttributes() - 1);

            //Build classifier with intervalInstances.
            if(classifier instanceof RandomTree){
                ((RandomTree)classifier).setKValue(intervalInstances.numAttributes() - 1);
            }
            baseClassifiers.add(AbstractClassifier.makeCopy(classifier));
            baseClassifiers.get(baseClassifiers.size()-1).buildClassifier(intervalInstances);

            //Add dependant variable to model (time taken).
            timer.dependantVariables.add(System.nanoTime() - timer.treeStartTime);

            //Serialise every 100 trees (if path has been set).
            if(treeCount % 100 == 0 && treeCount != 0 && serialisePath != null){
                saveToFile(String.valueOf(seed));
            }
        }
        if (serialisePath != null) {
            saveToFile(String.valueOf(seed));
        }

        if (timer.modelOutPath != null) {
            timer.saveModelToCSV(trainingData.relationName());
        }
        super.trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        super.trainResults.setBuildTime(System.nanoTime() - timer.forestStartTime);
    }

    public double[] distributionForInstance(Instance testInstance) throws java.lang.Exception {
        double[]distribution = new double[testInstance.numClasses()];
        for (int i = 0; i < baseClassifiers.size(); i++) {
            Instance intervalInstance = null;
            //Transform interval instance into PS, ACF, ACF_PS or ACF_PS_AR
            if (transformType != null) {
                try{
                    intervalInstance = catch22TransformInstances(transformInstances(produceIntervalInstanceUpdate(testInstance, i), transformType)).firstInstance();
                }catch(java.lang.Exception e){
                    intervalInstance = catch22TransformInstances(transformInstances(produceIntervalInstanceUpdate(testInstance, i), transformType)).firstInstance();
                }
            }
            distribution[(int)baseClassifiers.get(i).classifyInstance((intervalInstance))]++;
        }
        for (int j = 0; j < testInstance.numClasses(); j++) {
            distribution[j] /= baseClassifiers.size();
        }
        return distribution;
    }

    private Instances catch22TransformInstances(Instances intervalInstances){
        ArrayList<double[]> catchData = new ArrayList<>();
        for (int i = 0; i < intervalInstances.size(); i++) {
            double[] temp = new double[intervalInstances.get(i).toDoubleArray().length - 1];
            for (int j = 0; j < intervalInstances.get(i).toDoubleArray().length - 1; j++) {
                temp[j] = intervalInstances.get(i).toDoubleArray()[j];
            }
            catchData.add(catch22.transform(temp, intervalInstances.get(i).classValue()));
        }

        ArrayList<Attribute> atts = new ArrayList<>();
        for (int i = 1; i <= 22; i++){
            atts.add(new Attribute("att" + i));
        }
        atts.add(data.classAttribute());
        Instances transformedData = new Instances("Catch22Transform", atts, data.numInstances());
        transformedData.setClassIndex(transformedData.numAttributes()-1);
        for (int i = 0; i < catchData.size(); i++) {
            transformedData.add(new DenseInstance(1, catchData.get(i).clone()));
        }

        return transformedData;
    }

    public static void main(String[] args){

        Instances dataTrain = loadDataNullable("Z:/ArchiveData/Univariate_arff" + "/" + DatasetLists.newProblems27[1] + "/" + DatasetLists.newProblems27[1] + "_TRAIN");
        Instances dataTest = loadDataNullable("Z:/ArchiveData/Univariate_arff" + "/" + DatasetLists.newProblems27[1] + "/" + DatasetLists.newProblems27[1] + "_TEST");
        Instances data = dataTrain;
        data.addAll(dataTest);

        ClassifierResults cr = null;
        SingleSampleEvaluator sse = new SingleSampleEvaluator();
        sse.setPropInstancesInTrain(0.5);
        sse.setSeed(1);

        Catch22RISE Catch22RISE = null;
        System.out.println("Dataset name: " + data.relationName());
        System.out.println("Numer of cases: " + data.size());
        System.out.println("Number of attributes: " + (data.numAttributes() - 1));
        System.out.println("Number of classes: " + data.classAttribute().numValues());
        System.out.println("\n");
        try {
            Catch22RISE = new Catch22RISE();
            //cRISE.setTrainTimeLimit(TimeUnit.MINUTES, 5);
            Catch22RISE.setTransformType(TransformType.ACF);
            cr = sse.evaluate(Catch22RISE, data);
            System.out.println("ACF");
            System.out.println("Accuracy: " + cr.getAcc());
            System.out.println("Build time (ns): " + cr.getBuildTimeInNanos());

            Catch22RISE = new Catch22RISE();
            //cRISE.setSavePath("D:/Test/Testing/Serialising/");
            //cRISE.setTrainTimeLimit(TimeUnit.MINUTES, 5);
            Catch22RISE.setTransformType(TransformType.FFT);
            cr = sse.evaluate(Catch22RISE, data);
            System.out.println("FFT");
            System.out.println("Accuracy: " + cr.getAcc());
            System.out.println("Build time (ns): " + cr.getBuildTimeInNanos());

            Catch22RISE = new Catch22RISE();
            //cRISE.setSavePath("D:/Test/Testing/Serialising/");
            //cRISE.setTrainTimeLimit(TimeUnit.MINUTES, 5);
            Catch22RISE.setTransformType(TransformType.ACF_FFT);
            cr = sse.evaluate(Catch22RISE, data);
            System.out.println("ACF_FFT");
            System.out.println("Accuracy: " + cr.getAcc());
            System.out.println("Build time (ns): " + cr.getBuildTimeInNanos());
        } catch (java.lang.Exception e) {
            e.printStackTrace();
        }

/*        Catch22RISE = new Catch22RISE();
        try {
            ClassifierResults temp = ClassifierTools.testUtils_evalOnIPD(Catch22RISE);
            temp.writeFullResultsToFile("D:\\Test\\Testing\\TestyStuff\\cRISE.csv");
        } catch (Exception e) {
            e.printStackTrace();
        }*/
    }
}
