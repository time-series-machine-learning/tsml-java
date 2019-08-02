/*
 * Copyright (C) 2019 xmw13bzu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package weka_uea.classifiers.ensembles;

import evaluation.evaluators.CrossValidationEvaluator;
import timeseriesweka.classifiers.dictionary_based.BOSS;
import timeseriesweka.classifiers.distance_based.ElasticEnsemble;
import timeseriesweka.classifiers.frequency_based.RISE;
import timeseriesweka.classifiers.hybrids.HiveCote.DefaultShapeletTransformPlaceholder;
import timeseriesweka.classifiers.interval_based.TSF;
import weka.classifiers.Classifier;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka_uea.classifiers.ensembles.voting.MajorityConfidence;
import weka_uea.classifiers.ensembles.weightings.TrainAcc;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class HIVE_COTE extends AbstractEnsemble  implements TechnicalInformationHandler {

        @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "J. Lines, S. Taylor and A. Bagnall");
        result.setValue(TechnicalInformation.Field.TITLE, "Time Series Classification with HIVE-COTE: The Hierarchical Vote Collective of Transformation-Based Ensembles");
        result.setValue(TechnicalInformation.Field.JOURNAL, "ACM Transactions on Knowledge Discovery from Data");
        result.setValue(TechnicalInformation.Field.VOLUME, "12");
        result.setValue(TechnicalInformation.Field.NUMBER, "5");
        
        result.setValue(TechnicalInformation.Field.PAGES, "52");
        result.setValue(TechnicalInformation.Field.YEAR, "2018");
        return result;
    }    

    public HIVE_COTE() { 
        super();
    }
    
    @Override
    public void setupDefaultEnsembleSettings() {
        //copied over/adapted from HiveCote.setDefaultEnsembles()
        //for review purposes
        this.ensembleName = "HIVE-COTE";
        
        this.weightingScheme = new TrainAcc(4);
        this.votingScheme = new MajorityConfidence();
        this.transform = null;
        
        CrossValidationEvaluator cv = new CrossValidationEvaluator(seed, false, false, false, false); 
        cv.setNumFolds(10);
        this.trainEstimator = cv; 

        Classifier[] classifiers = new Classifier[5];
        String[] classifierNames = new String[5];
        
        classifiers[0] = new ElasticEnsemble();
        classifierNames[0] = "EE";
        
        CAWPE st_classifier = new CAWPE();
        DefaultShapeletTransformPlaceholder st_transform= new DefaultShapeletTransformPlaceholder();
        st_classifier.setTransform(st_transform);
        classifiers[1] = st_classifier;
        classifierNames[1] = "ST";
        
        classifiers[2] = new RISE();
        classifierNames[2] = "RISE";
        
        classifiers[3] = new BOSS();
        classifierNames[3] = "BOSS";
        
        classifiers[4] = new TSF();
        classifierNames[4] = "TSF";
        
        try {
            setClassifiers(classifiers, classifierNames, null);
        } catch (Exception e) {
            System.out.println("Exception thrown when setting up DEFUALT settings of " + this.getClass().getSimpleName() + ". Should "
                    + "be fixed before continuing");
            System.exit(1);
        }
        
        
        //defaults to 7 day contract TODO jay/tony review
        setTrainTimeLimit(contractTrainTimeUnit, contractTrainTime);
    }

}
