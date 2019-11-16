package machine_learning.classifiers.ensembles;

import evaluation.storage.ClassifierResults;
import machine_learning.classifiers.ensembles.weightings.TrainAcc;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Random;

import static utilities.Utilities.argMax;

public class HomogeneousContractCAWPE extends CAWPE {

    Random rand = new Random();

    public void setRandom(Random rand){
        this.rand = rand;
    }

    public void addToEnsemble(Classifier c, double[][] probs, double[] classVals) throws Exception {
        modules = Arrays.copyOf(modules, modules.length + 1);
        int idx = modules.length-1;
        modules[idx] = new EnsembleModule(c.getClass().getSimpleName(), c, "");

        modules[idx].trainResults = new ClassifierResults(numClasses);
        modules[idx].trainResults.setClassifierName(c.getClass().getSimpleName());
        modules[idx].trainResults.setDatasetName(trainInsts.relationName());
        modules[idx].trainResults.setFoldID(seed);
        modules[idx].trainResults.setSplit("train");

        for (int i = 0; i < probs.length; i++){
            double pred = argMax(probs[i], rand);
            modules[idx].trainResults.addPrediction(probs[i], pred, -1, "");
        }

        modules[idx].trainResults.finaliseResults(classVals);

        if (weightingScheme instanceof TrainAcc){
            modules[idx].posteriorWeights = ((TrainAcc)weightingScheme).defineWeighting(modules[idx], numClasses);
            votingScheme.trainVotingScheme(modules, numClasses);
        }
        else{
            weightingScheme.defineWeightings(modules, numClasses);
            votingScheme.trainVotingScheme(modules, numClasses);
        }

        modules[idx].trainResults.findAllStatsOnce();
    }

    public void remove(int idx){
        EnsembleModule[] temp = new EnsembleModule[modules.length - 1];
        System.arraycopy(modules, 0, temp, 0, idx);
        System.arraycopy(modules, idx + 1, temp, idx, modules.length - idx - 1);
        modules = temp;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        trainInsts = data;
        numTrainInsts = trainInsts.numInstances();
        numClasses = trainInsts.numClasses();
        numAttributes = trainInsts.numAttributes();
        modules = new EnsembleModule[0];
        this.testInstCounter = 0;
    }
}
