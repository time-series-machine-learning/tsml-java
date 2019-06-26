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

import static CawpeExtensionPaper.CAWPEClassifierList.datasetList;
import static CawpeExtensionPaper.CAWPEClassifierList.replaceLabelsForImages;
import evaluation.MultipleClassifierEvaluation;
import static CawpeExtensionPaper.CAWPEClassifierList.cawpeConfigs;

/**
 * Analysis setups to create/collate the results reported in the alcohol paper
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class CAWPEAnalysis {
    static String datasetPath = "C:/UCI Problems/";
    static String analysisPath = "C:/JamesLPHD/CAWPEExtension/Analysis/";
    static String resultsPath = "C:/JamesLPHD/CAWPEExtension/Results/";
            
    static int masterNumFolds = 30;
    
    public static void main(String[] args) throws Exception {
        ana_firsPass();
    }
    
    public static void ana_firsPass() throws Exception { 
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(analysisPath, "firstPass_diagrams", masterNumFolds);
        mce.setTestResultsOnly(true);
        mce.setBuildMatlabDiagrams(true);
        mce.setDatasets(datasetList);
        mce.setUseDefaultEvaluationStatistics();
        mce.readInClassifiers(cawpeConfigs, replaceLabelsForImages(cawpeConfigs), resultsPath);
        
        mce.runComparison();
    }
    
}
