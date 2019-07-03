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

package CawpeExtensionPaper;

import static CawpeExtensionPaper.CAWPEClassifierList.allBaseClassifiers;
import static CawpeExtensionPaper.CAWPEClassifierList.datasetList;
import static CawpeExtensionPaper.CAWPEClassifierList.replaceLabelsForImages;
import evaluation.MultipleClassifierEvaluation;
import static CawpeExtensionPaper.CAWPEClassifierList.cawpeConfigs;
import static CawpeExtensionPaper.CAWPEClassifierList.allTopLevelClassifiers;

/**
 * Analysis setups to create/collate the results reported in the CAWPE extension paper
 * studying the effects of maintaining the CV-fold models
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class CAWPEAnalysis {
    static String datasetPath = "C:/UCI Problems/";
    static String analysisPath = "C:/JamesLPHD/CAWPEExtension/Analysis/";
    static String resultsPath = "C:/JamesLPHD/CAWPEExtension/Results/";
            
    static int masterNumFolds = 30;
    
    public static void main(String[] args) throws Exception {
        ana_cawpeConfigsANDHomogeneous();
        ana_memberTrainTestDiffs();
    }
    
    public static void ana_cawpeConfigsANDHomogeneous() throws Exception { 
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(analysisPath, "ana_cawpeConfigsANDHomogeneous", masterNumFolds);
        mce.setTestResultsOnly(true);
        mce.setBuildMatlabDiagrams(true);
        mce.setDatasets(datasetList);
        mce.setUseDefaultEvaluationStatistics();
        mce.readInClassifiers(allTopLevelClassifiers, replaceLabelsForImages(allTopLevelClassifiers), resultsPath);
        
        mce.runComparison();
    }
    
    public static void ana_memberTrainTestDiffs() throws Exception { 
        
        // LONG EXECUTION TIME, TRAIN/TEST RESULTS FOR ALL INIDIVIDUALS AND FOLD CLASSIFIERS (i.e. 55 sets of results to load)
        
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(analysisPath, "ana_memberTrainTestDiffs", masterNumFolds);
        mce.setTestResultsOnly(false); // FALSE, want to compare train/test diffs of members
        mce.setBuildMatlabDiagrams(false);
        mce.setDatasets(datasetList);
        mce.setUseDefaultEvaluationStatistics();
        mce.readInClassifiers(allBaseClassifiers, replaceLabelsForImages(allBaseClassifiers), resultsPath);
        
        mce.runComparison();
    }
}
