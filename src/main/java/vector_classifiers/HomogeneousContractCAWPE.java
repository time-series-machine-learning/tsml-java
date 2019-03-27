package vector_classifiers;

import evaluation.storage.ClassifierResults;
import timeseriesweka.classifiers.ensembles.EnsembleModule;
import timeseriesweka.classifiers.ensembles.weightings.TrainAcc;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.util.Arrays;

public class HomogeneousContractCAWPE extends CAWPE {

    public void addToEnsemble(Classifier c, double[][] probs, double[] classVals) throws Exception {
        Arrays.copyOf(modules,modules.length+1);
        int idx = modules.length-1;
        modules[idx] = new EnsembleModule(c.getClass().getSimpleName(), c, "");

        modules[idx].trainResults = new ClassifierResults(numClasses);
        modules[idx].trainResults.setClassifierName(c.getClass().getSimpleName());
        modules[idx].trainResults.setDatasetName(trainInsts.relationName());
        modules[idx].trainResults.setFoldID(seed);
        modules[idx].trainResults.setSplit("train");

        for (int i = 0; i < probs.length; i++){
            double pred = 1;
            modules[idx].trainResults.addPrediction(probs[i], pred, -1, "");
        }

        modules[idx].trainResults.finaliseResults(classVals);

        if (weightingScheme instanceof TrainAcc){
            ((TrainAcc)weightingScheme).defineWeighting(modules[idx], numClasses);
            votingScheme.trainVotingScheme(modules, numClasses);
        }
        else{
            weightingScheme.defineWeightings(modules, numClasses);
            votingScheme.trainVotingScheme(modules, numClasses);
        }

        modules[idx].trainResults.findAllStatsOnce();
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
