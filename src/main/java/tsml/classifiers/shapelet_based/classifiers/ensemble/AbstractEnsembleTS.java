package tsml.classifiers.shapelet_based.classifiers.ensemble;


import evaluation.storage.ClassifierResults;
import tsml.classifiers.TSClassifier;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import utilities.DebugPrinting;

import java.util.Random;

public abstract class AbstractEnsembleTS implements TSClassifier {

    //Main ensemble design decisions/variables
    protected String ensembleName;
    protected ModuleWeightingSchemeTS weightingScheme;
    protected ModuleVotingSchemeTS votingScheme;
    protected EnsembleModuleTS[] modules;
    protected int numEnsembles = 100;

    protected long buildTime = -1;


    protected Random rand = new Random();

    public AbstractEnsembleTS() {

    }



    public abstract void setupDefaultEnsembleSettings(TimeSeriesInstances data);


    public static class EnsembleModuleTS implements DebugPrinting {
        private TSClassifier classifier;

        public ClassifierResults trainResults;
        public ClassifierResults testResults;

        public double priorWeight = 1.0;
        public double[] posteriorWeights;


        public EnsembleModuleTS(TSClassifier classifier) {
            this.classifier = classifier;
        }

        public TSClassifier getClassifier() {
            return classifier;
        }

        public void setClassifier(TSClassifier classifier) {
            this.classifier = classifier;
        }

        @Override
        public String toString() {
            return classifier.toString();
        }
    }






    public TSClassifier[] getClassifiers(){
        TSClassifier[] classifiers = new TSClassifier[modules.length];
        for (int i = 0; i < modules.length; i++)
            classifiers[i] = modules[i].getClassifier();
        return classifiers;
    }



    public void setClassifiers(TSClassifier[] classifiers) {

        this.modules = new EnsembleModuleTS[classifiers.length];
        for (int m = 0; m < modules.length; m++)
            modules[m] = new EnsembleModuleTS(classifiers[m]);
    }








    public double[][] getPosteriorIndividualWeights() {
        double[][] weights = new double[modules.length][];
        for (int m = 0; m < modules.length; ++m)
            weights[m] = modules[m].posteriorWeights;

        return weights;
    }

    public double[] getIndividualAccEstimates() {
        double [] accs = new double[modules.length];
        for (int i = 0; i < modules.length; i++)
            accs[i] = modules[i].trainResults.getAcc();

        return accs;
    }


    public EnsembleModuleTS[] getModules() {
        return modules;
    }



    public String getEnsembleName() {
        return ensembleName;
    }
    public void setEnsembleName(String ensembleName) {
        this.ensembleName = ensembleName;
    }


    public ModuleVotingSchemeTS getVotingScheme() {
        return votingScheme;
    }

    public void setVotingScheme(ModuleVotingSchemeTS votingScheme) {
        this.votingScheme = votingScheme;
    }

    public ModuleWeightingSchemeTS getWeightingScheme() {
        return weightingScheme;
    }

    public void setWeightingScheme(ModuleWeightingSchemeTS weightingScheme) {
        this.weightingScheme = weightingScheme;
    }


    public double[] getPriorIndividualWeights() {
        double[] priors = new double[modules.length];
        for (int i = 0; i < modules.length; i++)
            priors[i] = modules[i].priorWeight;

        return priors;
    }

    public void setPriorIndividualWeights(double[] priorWeights) throws Exception {
        if (priorWeights.length != modules.length)
            throw new Exception("Number of prior weights being set (" + priorWeights.length
                    + ") not equal to the number of modules (" + modules.length + ")");

        for (int i = 0; i < modules.length; i++)
            modules[i].priorWeight = priorWeights[i];
    }

    private void setDefaultPriorWeights() {
        for (int i = 0; i < modules.length; i++)
            modules[i].priorWeight = 1.0;
    }

    public double[][] getIndividualEstimatePredictions() {
        double [][] preds = new double[modules.length][];
        for (int i = 0; i < modules.length; i++)
            preds[i] = modules[i].trainResults.getPredClassValsAsArray();
        return preds;
    }





    @Override
    public void buildClassifier(TimeSeriesInstances data) throws Exception {
        setupDefaultEnsembleSettings(data);

        long startTime = System.nanoTime();



        //set up ensemble
        weightingScheme.defineWeightings(modules, data.numClasses());
        votingScheme.trainVotingScheme(modules, data.numClasses());
        for (int i=0;i<numEnsembles;i++){

            modules[i].classifier.buildClassifier(data);
        }
    }

    @Override
    public double[] distributionForInstance(TimeSeriesInstance data) throws Exception {
        return votingScheme.distributionForInstance(modules, data);
    }

    @Override
    public double classifyInstance(TimeSeriesInstance data) throws Exception {
        return votingScheme.classifyInstance(modules,data);
    }

}
