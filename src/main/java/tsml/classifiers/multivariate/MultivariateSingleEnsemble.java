package tsml.classifiers.multivariate;

import development.CIF;
import development.TDE;
import evaluation.evaluators.CrossValidationEvaluator;
import machine_learning.classifiers.ensembles.voting.MajorityConfidence;
import machine_learning.classifiers.ensembles.weightings.EqualWeighting;
import machine_learning.classifiers.ensembles.weightings.TrainAcc;
import tsml.classifiers.dictionary_based.cBOSS;
import tsml.classifiers.frequency_based.RISE;
import tsml.classifiers.hybrids.HIVE_COTE;
import tsml.classifiers.interval_based.TSF;
import tsml.classifiers.shapelet_based.ShapeletTransformClassifier;
import weka.classifiers.Classifier;

public class MultivariateSingleEnsemble extends MultivariateAbstractEnsemble {

    private String resultsPath;
    private String dataset;
    private int fold;
    private String classifierName;

    public MultivariateSingleEnsemble(String classifierName, String resultsPath, String dataset, int fold) {
        this.classifierName = classifierName;
        this.resultsPath = resultsPath;
        this.dataset = dataset;
        this.fold = fold;
        this.setBuildIndividualsFromResultsFiles(true);
        this.setResultsFileLocationParameters(resultsPath,dataset, fold);
        setSeed(fold);
        setDebug(true);
    }

    @Override
    protected void setupMultivariateEnsembleSettings(int instancesLength) {
            this.ensembleName = "MTSC_"+ this.classifierName +"_I";
//            this.weightingScheme = new EqualWeighting();
//            this.votingScheme = new MajorityConfidence();
        this.weightingScheme = new TrainAcc(4);
        this.votingScheme = new MajorityConfidence();
        this.transform = null;
        CrossValidationEvaluator cv = new CrossValidationEvaluator(seed, false, false, false, false);
        cv.setNumFolds(10);
        this.trainEstimator = cv;

        Classifier[] classifiers = new Classifier[instancesLength];
        String[] classifierNames = new String[instancesLength];

        for (int i=0;i<instancesLength;i++){

            classifiers[i] = getClassifier();
            classifierNames[i] = this.classifierName;
        }


        setClassifiers(classifiers, classifierNames, null);

    }

    private Classifier getClassifier(){
        switch (this.classifierName){
            case "cBOSS": return new cBOSS();
            case "RISE": return new RISE();
            case "STC": return new ShapeletTransformClassifier();
            case "TSF": return new TSF();
            case "TDE": return new TDE();
            case "CIF": return new CIF();
            default: return new TSF();

        }
    }
}
