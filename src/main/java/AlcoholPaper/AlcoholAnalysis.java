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

package AlcoholPaper;

import evaluation.MultipleClassifierEvaluation;

/**
 * Analysis setups to create/collate the results reported in the alcohol paper
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class AlcoholAnalysis {
    
    static String datasetPath = "C:\\JamesLPHD\\Alcohol\\JOURNALPAPER\\Datasets\\";
    static String analysisPath = "C:\\JamesLPHD\\Alcohol\\JOURNALPAPER\\Analysis\\";
    static String resultsPath = "C:\\JamesLPHD\\Alcohol\\JOURNALPAPER\\Results\\";
            
    
    public static void main(String[] args) {
        
    }
    
    public static void ana_alcConc_LOBO() throws Exception { 
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(analysisPath, "alcConc_LOBO", 44);
        mce.setTestResultsOnly(true);
        mce.setBuildMatlabDiagrams(false);
        mce.setDatasets(new String[] { "AlcoholForgeryEthanol", "AlcoholForgeryMethanol" });
        mce.setUseAllStatistics();
        mce.readInClassifiers(AlcoholClassifierList.classifiers_all, resultsPath);
        
        mce.runComparison();
    }
    
    public static void ana_alcConc_LOBOPCA() throws Exception { 
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(analysisPath, "alcConc_LOBOPCA", 44);
        mce.setTestResultsOnly(true);
        mce.setBuildMatlabDiagrams(false);
        mce.setDatasets(new String[] { "PCAAlcoholForgeryEthanol", "PCAAlcoholForgeryMethanol" });
        mce.setUseAllStatistics();
        mce.readInClassifiers(AlcoholClassifierList.classifiers_nonTsc, resultsPath);
        
        mce.runComparison();
    }
    
    public static void ana_alcConc_RandomBottle() throws Exception { 
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(analysisPath, "alcConc_LOBOPCA", 30);
        mce.setTestResultsOnly(true);
        mce.setBuildMatlabDiagrams(false);
        mce.setDatasets(new String[] { "RandomBottlesEthanol", });
        mce.setUseAllStatistics();
        mce.readInClassifiers(AlcoholClassifierList.classifiers_nonTsc, resultsPath);
        
        mce.runComparison();
    }
    
    
    public static void ana_jw_30RandFold() throws Exception { 
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(analysisPath, "jw_30RandFold", 30);
        mce.setTestResultsOnly(true);
        mce.setBuildMatlabDiagrams(false);
        mce.setDatasets(new String[] { "JWRorJWB_RedBottle", "JWRorJWB_BlackBottle", });
        mce.setUseAllStatistics();
        mce.readInClassifiers(AlcoholClassifierList.classifiers_all, resultsPath);
        
        mce.runComparison();
    }
    
    
    public static void ana_jw_1FoldUSerSplit() throws Exception { 
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(analysisPath, "jw_FoldUSerSplit", 1);
        mce.setTestResultsOnly(true);
        mce.setBuildMatlabDiagrams(false);
        mce.setDatasets(new String[] { "JWRorJWB_RedBottle", "JWRorJWB_BlackBottle", });
        mce.setUseAllStatistics();
        mce.readInClassifiers(AlcoholClassifierList.classifiers_all, resultsPath);
        
        mce.runComparison();
    }
}
