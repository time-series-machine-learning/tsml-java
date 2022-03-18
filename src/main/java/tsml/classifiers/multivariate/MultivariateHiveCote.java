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
 
package tsml.classifiers.multivariate;

import evaluation.evaluators.CrossValidationEvaluator;
import machine_learning.classifiers.ensembles.voting.MajorityConfidence;
import machine_learning.classifiers.ensembles.weightings.EqualWeighting;
import machine_learning.classifiers.ensembles.weightings.TrainAcc;
import tsml.classifiers.hybrids.HIVE_COTE;
import weka.classifiers.Classifier;
import weka.core.Capabilities;

public class MultivariateHiveCote extends MultivariateAbstractEnsemble {

    private String resultsPath;
    private String dataset;
    private int fold;

    public MultivariateHiveCote(String resultsPath, String dataset, int fold) {
        this.resultsPath = resultsPath;
        this.dataset = dataset;
        this.fold = fold;
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        // attributes must be numeric
        // Here add in relational when ready
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_VALUES);
        // instances
        result.setMinimumNumberInstances(1);
        return result;
    }

    @Override
    protected void setupMultivariateEnsembleSettings(int instancesLength) {
            this.ensembleName = "MTSC_HC_I";

        this.weightingScheme = new TrainAcc(4);
        this.votingScheme = new MajorityConfidence();
            this.transform = null;

            CrossValidationEvaluator cv = new CrossValidationEvaluator(seed, false, false, false, false);
            cv.setNumFolds(10);
            this.trainEstimator = cv;

            Classifier[] classifiers = new Classifier[instancesLength];
            String[] classifierNames = new String[instancesLength];

            for (int i=0;i<instancesLength;i++){
                String[] cls={"TSF","cBOSS","RISE","STC"};
                classifiers[i] = new HIVE_COTE();
                ((HIVE_COTE)classifiers[i]).setFillMissingDistsWithOneHotVectors(true);
                ((HIVE_COTE)classifiers[i]).setSeed(fold);
                ((HIVE_COTE)classifiers[i]).setBuildIndividualsFromResultsFiles(true);
                ((HIVE_COTE)classifiers[i]).setResultsFileLocationParameters(resultsPath, dataset+"Dimension"+(i+1), fold);
                ((HIVE_COTE)classifiers[i]).setClassifiersNamesForFileRead(cls);
                classifierNames[i] = "HC-Channel-" + (i+1);
            }


            setClassifiers(classifiers, classifierNames, null);

    }
}
