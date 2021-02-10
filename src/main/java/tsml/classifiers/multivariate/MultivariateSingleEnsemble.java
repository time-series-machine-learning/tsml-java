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
import machine_learning.classifiers.ensembles.weightings.TrainAcc;
import tsml.classifiers.dictionary_based.cBOSS;
import tsml.classifiers.interval_based.RISE_KNNProxy;
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

            classifiers[i] = getClassifierFromString();
            classifierNames[i] = this.classifierName;
        }


        setClassifiers(classifiers, classifierNames, null);

    }

    private Classifier getClassifierFromString(){
        switch (this.classifierName){
            case "cBOSS": return new cBOSS();
            case "RISE": return new RISE_KNNProxy();
            case "STC": return new ShapeletTransformClassifier();
            case "TSF": return new TSF();
            default: return new TSF();

        }
    }
}
