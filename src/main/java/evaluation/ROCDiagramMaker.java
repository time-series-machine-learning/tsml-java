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

package evaluation;

import ResultsProcessing.MatlabController;
import static evaluation.ClassifierResultsAnalysis.matlabFilePath;
import evaluation.storage.ClassifierResults;
import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Class to convert results in the format as they are in the results pipeline (ClassifierResults) 
 * into the format for the roccurves.m matlab script for generating roc diagrams
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class ROCDiagramMaker {

    public static String rocDiaPath = "dias_ROCCurve/";
    
    public static String[] formatClassifierNames(String[] cnames) { 
        int maxLength = -1; 
        for (String cname : cnames)
            if (cname.length() > maxLength)
                maxLength = cname.length();
        String[] paddedNames = new String[cnames.length];
        for (int i = 0; i < cnames.length; i++) {
            paddedNames[i] = cnames[i];
            while(paddedNames[i].length() < maxLength)
                paddedNames[i] += " ";
        }
        return paddedNames;
    }
    
    public static double[] extractPosClassProbabilities(ClassifierResults results, int positiveClass) { 
        double[][] dists = results.getProbabilityDistributionsAsArray();
        double[] posClassProbs = new double[dists.length];
        
        for (int i = 0; i < posClassProbs.length; i++)
            posClassProbs[i] = dists[i][positiveClass];
        
        return posClassProbs;
    }
    
    public static int findMinorityClass(double[] classVals) { 
        HashMap<Integer,Integer> classes = new HashMap<>();
        
        for (double classVal : classVals) {
            Integer v = classes.get(classVal);
            if (v == null) 
                v = 0;
            classes.put((int)classVal, v++);
        }
        
        int minClass = -1, minCount = Integer.MAX_VALUE;
        for (Map.Entry<Integer, Integer> entry : classes.entrySet()) {
            if (entry.getValue() < minCount) {
                minCount = entry.getValue();
                minClass = entry.getKey();
            }
        }
        
        return minClass;
    }
        
    public static void matlab_buildROCDiagrams(String outPath, String expName, String dsetName, ClassifierResults[] cresults, String[] cnames) {   
        matlab_buildROCDiagrams(outPath, expName, dsetName, cresults, cnames, findMinorityClass(cresults[0].getTrueClassValsAsArray()));
    }   
    
    public static void matlab_buildROCDiagrams(String outPath, String expName, String dsetName, ClassifierResults[] cresults, String[] cnames, int positiveClassIndex) {      
        String targetFolder = outPath + rocDiaPath;
        (new File(targetFolder)).mkdirs();
        
        String targetFile = targetFolder + "rocDia_" + expName + "_" + dsetName;
        
        try {
            MatlabController proxy = MatlabController.getInstance();

            proxy.eval("addpath(genpath('"+matlabFilePath+"'))");
            
            proxy.eval("m_fname = '" + targetFile + "';");
            
            //holy hacks batman
            // turns [CAWPE, resnet, XGBoost] 
            // into  ['CAWPE  '; 'resnet '; 'XGBoost']
            String[] paddedNames = formatClassifierNames(cnames);
//            System.out.println("m_cnames = " + Arrays.toString(paddedNames).replace(", ", "'; '").replace("[", "['").replace("]", "']") + "");
            proxy.eval("m_cnames = " + Arrays.toString(paddedNames).replace(", ", "'; '").replace("[", "['").replace("]", "']") + ";");
            
            double[] cvals = cresults[0].getTrueClassValsAsArray();
            int[] m_cvals = new int[cvals.length];
            for (int i = 0; i < cvals.length; i++)
                m_cvals[i] = (int)cvals[i];
            proxy.eval("m_cvals = " + Arrays.toString(m_cvals) + ";");

            StringBuilder probsSB = new StringBuilder();
            for (int i = 0; i < cresults.length; i++)  {
                double[] probs = extractPosClassProbabilities(cresults[i], positiveClassIndex);
                probsSB.append(Arrays.toString(probs).replace("[", "").replace("]", ";"));
            }
            proxy.eval("m_posClassProbs = [ " + probsSB.toString() + " ];");
            
            proxy.eval("m_posClass = " + positiveClassIndex + ";");
            
            //function [f] = roccurves(filepathandname,classifierNames,classValues,posClassProbs,posClassLabel,visible)
            proxy.eval("roccurves(m_fname, m_cnames, m_cvals, m_posClassProbs, m_posClass, 'off')");
            proxy.eval("clear");
            proxy.discconnectMatlab();
        } catch (Exception io) {
            System.out.println("matlab_buildROCDiagrams failed while building " +targetFile+ "\n" + io);
        }
    }
    
    public static void main(String[] args) throws Exception {
        
        String baseReadPath = "C:/JamesLPHD/Alcohol/JOURNALPAPER/Results/";
        String dset = "JWRorJWB_BlackBottle";
        String[] cnames = { "CAWPE", "resnet", "XGBoost" };
        int numFolds = 10;
        
        ClassifierResults[][] res = new ClassifierResults[cnames.length][numFolds];
        for (int i = 0; i < res.length; i++) {
            for (int f = 0; f < numFolds; f++) {
                res[i][f] = new ClassifierResults(baseReadPath + cnames[i] + "/Predictions/" + dset + "/testFold"+f+".csv");
            }
        }
        
        ClassifierResults[] concatenatedRes = ClassifierResults.concatenateClassifierResults(res);
        matlab_buildROCDiagrams("C:/Temp/rocDiaTest/", "testDias", dset, concatenatedRes, cnames);
        
        //single fold 
//        String baseReadPath = "C:/JamesLPHD/Alcohol/JOURNALPAPER/Results/";
//        String dset = "JWRorJWB_BlackBottle";
//        String[] cnames = { "CAWPE", "resnet", "XGBoost" }; 
//        
//        ClassifierResults[] res = new ClassifierResults[cnames.length];
//        for (int i = 0; i < res.length; i++)
//            res[i] = new ClassifierResults(baseReadPath + cnames[i] + "/Predictions/" + dset + "/testFold0.csv");
//        
//        matlab_buildROCDiagrams("C:/Temp/rocDiaTest/", "testDias", dset, res, cnames);
    }
    
    
}
