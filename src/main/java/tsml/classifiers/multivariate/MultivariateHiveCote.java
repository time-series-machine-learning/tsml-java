package tsml.classifiers.multivariate;

import evaluation.evaluators.CrossValidationEvaluator;
import machine_learning.classifiers.ensembles.voting.MajorityConfidence;
import machine_learning.classifiers.ensembles.voting.MultivariateMajorityConfidence;
import machine_learning.classifiers.ensembles.weightings.TrainAcc;
import tsml.classifiers.dictionary_based.BOSS;
import weka.classifiers.Classifier;

public class MultivariateHiveCote extends MultivariateAbstractEnsemble {

    @Override
    protected void setupMultivariateEnsembleSettings(int instancesLength) {
            this.ensembleName = "MTSC_HC_I";

            this.weightingScheme = new TrainAcc(4);
            this.votingScheme = new MultivariateMajorityConfidence();
            this.transform = null;

            CrossValidationEvaluator cv = new CrossValidationEvaluator(seed, false, false, false, false);
            cv.setNumFolds(10);
            this.trainEstimator = cv;

            Classifier[] classifiers = new Classifier[instancesLength];
            String[] classifierNames = new String[instancesLength];

            for (int i=0;i<instancesLength;i++){
                classifiers[i] = new BOSS();
                classifierNames[i] = "HC-Channel-" + (i+1);
            }


            setClassifiers(classifiers, classifierNames, null);

    }
}
